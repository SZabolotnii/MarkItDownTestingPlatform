"""Core processing logic for the MarkItDown Testing Platform."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import JsonValue

from core.modules import (
    HFConversionEngine,
    ProcessingConfig,
    ProcessingResult,
    ResourceManager,
    StreamlineFileHandler,
)
from llm.gemini_connector import (
    AnalysisRequest,
    AnalysisType,
    GeminiConfig,
    GeminiConnectionManager,
    GeminiModel,
)
from visualization.analytics_engine import QualityMetricsCalculator


logger = logging.getLogger(__name__)

JSONDict = Dict[str, JsonValue]


@dataclass(frozen=True)
class ProcessingRequest:
    """Immutable request container describing a processing job."""

    file_content: bytes
    file_metadata: JSONDict
    gemini_api_key: Optional[str] = None
    analysis_type: str = AnalysisType.QUALITY_ANALYSIS.value
    model_preference: str = GeminiModel.PRO.value
    use_llm: bool = False
    enable_plugins: bool = False
    azure_endpoint: Optional[str] = None
    session_context: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResponse:
    """Standardized response describing the outcome of processing."""

    success: bool
    conversion_result: Optional[ProcessingResult]
    analysis_result: Optional[Any]
    quality_metrics: JSONDict
    error_details: Optional[str]
    processing_metadata: JSONDict

    @classmethod
    def success_response(
        cls,
        conversion_result: ProcessingResult,
        analysis_result: Optional[Any] = None,
        quality_metrics: Optional[JSONDict] = None,
    ) -> "ProcessingResponse":
        return cls(
            success=True,
            conversion_result=conversion_result,
            analysis_result=analysis_result,
            quality_metrics=quality_metrics or {},
            error_details=None,
            processing_metadata={"completed_at": datetime.now().isoformat()},
        )

    @classmethod
    def error_response(
        cls,
        error_message: str,
        error_context: Optional[JSONDict] = None,
    ) -> "ProcessingResponse":
        return cls(
            success=False,
            conversion_result=None,
            analysis_result=None,
            quality_metrics={},
            error_details=error_message,
            processing_metadata=error_context or {"failed_at": datetime.now().isoformat()},
        )


class DocumentProcessingOrchestrator:
    """Coordinates the document conversion and optional AI analysis pipeline."""

    def __init__(
        self,
        file_handler: StreamlineFileHandler,
        conversion_engine: HFConversionEngine,
        gemini_manager: GeminiConnectionManager,
        quality_calculator: QualityMetricsCalculator,
    ) -> None:
        self.file_handler = file_handler
        self.conversion_engine = conversion_engine
        self.gemini_manager = gemini_manager
        self.quality_calculator = quality_calculator

        self.processing_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

    async def process_document(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a document and optionally run Gemini analysis."""

        processing_start = datetime.now()
        self.processing_count += 1

        try:
            logger.info(
                "Starting document processing - Session: %s | LLM Enabled: %s",
                request.session_context.get("session_id", "unknown"),
                request.use_llm,
            )

            conversion_result = await self._execute_conversion_pipeline(request)
            if not conversion_result.success:
                return ProcessingResponse.error_response(
                    f"Conversion failed: {conversion_result.error_message}",
                    {"phase": "conversion", "request_metadata": request.file_metadata},
                )

            analysis_result = None
            if request.gemini_api_key:
                analysis_result = await self._execute_analysis_pipeline(request, conversion_result)

            quality_metrics = self.quality_calculator.calculate_conversion_quality_metrics(
                conversion_result, analysis_result
            )

            processing_duration = (datetime.now() - processing_start).total_seconds()
            self.total_processing_time += processing_duration

            logger.info("Processing completed successfully in %.2fs", processing_duration)

            return ProcessingResponse.success_response(
                conversion_result=conversion_result,
                analysis_result=analysis_result,
                quality_metrics=quality_metrics,
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.error_count += 1
            error_duration = (datetime.now() - processing_start).total_seconds()

            logger.error("Processing failed after %.2fs: %s", error_duration, exc)

            return ProcessingResponse.error_response(
                error_message=f"System processing error: {exc}",
                error_context={
                    "processing_duration": error_duration,
                    "error_type": type(exc).__name__,
                    "processing_phase": "unknown",
                },
            )

    async def _execute_conversion_pipeline(self, request: ProcessingRequest) -> ProcessingResult:
        """Handle file ingestion, validation, and conversion to Markdown."""

        class ProcessingFile:
            def __init__(self, content: bytes, metadata: JSONDict) -> None:
                self.content = content
                self.name = metadata.get("filename", "uploaded_file")

            def read(self) -> bytes:
                return self.content

            @property
            def size(self) -> int:
                return len(self.content)

        processing_file = ProcessingFile(request.file_content, request.file_metadata)

        file_result = await self.file_handler.process_upload(
            processing_file,
            metadata_override=request.file_metadata,
        )
        if not file_result.success:
            return file_result

        conversion_result = await self.conversion_engine.convert_stream(
            request.file_content,
            request.file_metadata,
        )

        return conversion_result

    async def _execute_analysis_pipeline(
        self,
        request: ProcessingRequest,
        conversion_result: ProcessingResult,
    ) -> Optional[Any]:
        """Run Gemini analysis with retry and error handling."""

        try:
            gemini_config = GeminiConfig(api_key=request.gemini_api_key)
            engine_id = await self.gemini_manager.create_engine(
                request.gemini_api_key,
                gemini_config,
            )

            engine = self.gemini_manager.get_engine(engine_id)
            if not engine:
                logger.warning("Gemini engine creation failed - skipping analysis")
                return None

            analysis_request = AnalysisRequest(
                content=conversion_result.content,
                analysis_type=AnalysisType(request.analysis_type),
                model=GeminiModel.from_str(request.model_preference),
            )

            analysis_result = await engine.analyze_content(analysis_request)
            if analysis_result.success:
                logger.info("Gemini analysis completed - Type: %s", request.analysis_type)
                return analysis_result

            logger.warning("Gemini analysis failed: %s", analysis_result.error_message)
            return None

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Gemini analysis pipeline error: %s", exc)
            return None

    def get_processing_status(self) -> JSONDict:
        """Expose operational metrics for status dashboards."""

        success_rate = (
            ((self.processing_count - self.error_count) / self.processing_count * 100)
            if self.processing_count
            else 0
        )

        average_processing_time = (
            self.total_processing_time / self.processing_count if self.processing_count else 0
        )

        return {
            "total_documents_processed": self.processing_count,
            "success_rate_percent": success_rate,
            "error_count": self.error_count,
            "average_processing_time_seconds": average_processing_time,
            "total_processing_time_seconds": self.total_processing_time,
            "status": "healthy"
            if success_rate > 90
            else "degraded"
            if success_rate > 70
            else "unhealthy",
        }


__all__ = [
    "JSONDict",
    "ProcessingRequest",
    "ProcessingResponse",
    "DocumentProcessingOrchestrator",
    "ProcessingConfig",
    "ResourceManager",
    "StreamlineFileHandler",
    "HFConversionEngine",
    "GeminiConnectionManager",
    "QualityMetricsCalculator",
]
