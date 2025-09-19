"""
MarkItDown Testing Platform - Enterprise Architecture Implementation

Strategic Design Philosophy:
"Complexity is the enemy of reliable software"

Core Architectural Principles:
- Minimize cognitive load for developers
- Create self-documenting, modular interfaces  
- Design for future adaptability
- Prioritize human understanding over technical complexity

This implementation demonstrates enterprise-grade architectural patterns
optimized for long-term maintainability and team collaboration.
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Protocol, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import gradio as gr
from pathlib import Path
from pydantic import JsonValue

# Strategic import organization - dependency layers clearly defined
from core.modules import (
    StreamlineFileHandler, HFConversionEngine, ResourceManager,
    ProcessingConfig, ProcessingResult
)
from llm.gemini_connector import (
    GeminiAnalysisEngine, GeminiConnectionManager, GeminiConfig,
    AnalysisRequest, AnalysisType, GeminiModel
)
from visualization.analytics_engine import (
    InteractiveVisualizationEngine, QualityMetricsCalculator,
    VisualizationConfig, ReportGenerator
)

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== SERIALIZABLE TYPE DEFINITIONS ====================

JSONDict = Dict[str, JsonValue]


if hasattr(gr.Blocks, "get_api_info"):
    def _suppress_api_info(self):
        return {"named_endpoints": {}, "unnamed_endpoints": []}

    gr.Blocks.get_api_info = _suppress_api_info


# ==================== STRATEGIC DATA MODELS ====================

@dataclass(frozen=True)
class ProcessingRequest:
    """Immutable request container - eliminates parameter coupling"""
    
    file_content: bytes
    file_metadata: JSONDict
    gemini_api_key: Optional[str] = None
    analysis_type: str = "quality_analysis"
    model_preference: str = GeminiModel.PRO.value
    enable_plugins: bool = False
    azure_endpoint: Optional[str] = None
    session_context: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResponse:
    """Standardized response container - predictable interface"""
    
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
        analysis_result: Any = None,
        quality_metrics: Optional[JSONDict] = None
    ) -> 'ProcessingResponse':
        """Factory method for successful processing"""
        return cls(
            success=True,
            conversion_result=conversion_result,
            analysis_result=analysis_result,
            quality_metrics=quality_metrics or {},
            error_details=None,
            processing_metadata={'completed_at': datetime.now().isoformat()}
        )
    
    @classmethod
    def error_response(cls, error_message: str, error_context: Optional[JSONDict] = None) -> 'ProcessingResponse':
        """Factory method for error scenarios"""
        return cls(
            success=False,
            conversion_result=None,
            analysis_result=None,
            quality_metrics={},
            error_details=error_message,
            processing_metadata=error_context or {'failed_at': datetime.now().isoformat()}
        )


@dataclass
class ApplicationState:
    """Centralized state management - eliminates state scatter"""
    
    session_id: str
    processing_history: List[ProcessingResponse] = field(default_factory=list)
    current_gemini_engine_id: Optional[str] = None
    user_preferences: JSONDict = field(default_factory=dict)
    system_metrics: JSONDict = field(default_factory=dict)
    
    def add_processing_result(self, response: ProcessingResponse) -> 'ApplicationState':
        """Immutable state update pattern"""
        new_history = self.processing_history + [response]
        return ApplicationState(
            session_id=self.session_id,
            processing_history=new_history,
            current_gemini_engine_id=self.current_gemini_engine_id,
            user_preferences=self.user_preferences,
            system_metrics=self.system_metrics
        )


# ==================== STRATEGIC ABSTRACTION LAYER ====================

class ProcessingOrchestrator(Protocol):
    """Interface abstraction - enables component replacement"""
    
    async def process_document(self, request: ProcessingRequest) -> ProcessingResponse:
        """Core processing contract"""
        ...
    
    def get_processing_status(self) -> JSONDict:
        """System health interface"""
        ...


class UIResponseFactory(Protocol):
    """UI generation abstraction - separates presentation from logic"""
    
    def create_success_response(self, response: ProcessingResponse) -> Tuple[str, str, str, JSONDict]:
        """Generate UI components for successful processing"""
        ...
    
    def create_error_response(self, error_message: str) -> Tuple[str, str, str, JSONDict]:
        """Generate UI components for error scenarios"""
        ...


# ==================== CORE ORCHESTRATION IMPLEMENTATION ====================

class DocumentProcessingOrchestrator:
    """
    Strategic orchestration layer - coordinates component interactions
    
    Design Principles:
    - Single Responsibility: Document processing coordination only
    - Dependency Injection: All components provided at construction
    - Error Boundary: Comprehensive error handling and recovery
    - Observable: Rich logging and metrics for operational visibility
    """
    
    def __init__(
        self,
        file_handler: StreamlineFileHandler,
        conversion_engine: HFConversionEngine,
        gemini_manager: GeminiConnectionManager,
        viz_engine: InteractiveVisualizationEngine,
        quality_calculator: QualityMetricsCalculator
    ):
        self.file_handler = file_handler
        self.conversion_engine = conversion_engine
        self.gemini_manager = gemini_manager
        self.viz_engine = viz_engine
        self.quality_calculator = quality_calculator
        
        # Operational metrics
        self.processing_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
    
    async def process_document(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Primary processing coordination with comprehensive error handling
        
        Strategic Approach:
        1. Input validation and sanitization
        2. Resource availability verification  
        3. Processing pipeline execution with checkpoints
        4. Quality assessment and metrics generation
        5. Response standardization and logging
        """
        
        processing_start = datetime.now()
        self.processing_count += 1
        
        try:
            logger.info(f"Starting document processing - Session: {request.session_context.get('session_id', 'unknown')}")
            
            # Phase 1: Document Ingestion and Validation
            conversion_result = await self._execute_conversion_pipeline(request)
            if not conversion_result.success:
                return ProcessingResponse.error_response(
                    f"Conversion failed: {conversion_result.error_message}",
                    {"phase": "conversion", "request_metadata": request.file_metadata}
                )
            
            # Phase 2: AI Analysis (Optional Enhancement)
            analysis_result = None
            if request.gemini_api_key:
                analysis_result = await self._execute_analysis_pipeline(
                    request, conversion_result
                )
                # Note: Analysis failure is non-fatal - system continues with conversion results
            
            # Phase 3: Quality Assessment and Metrics Generation
            quality_metrics = self.quality_calculator.calculate_conversion_quality_metrics(
                conversion_result, analysis_result
            )
            
            # Phase 4: Response Assembly and Logging
            processing_duration = (datetime.now() - processing_start).total_seconds()
            self.total_processing_time += processing_duration
            
            logger.info(f"Processing completed successfully in {processing_duration:.2f}s")
            
            return ProcessingResponse.success_response(
                conversion_result=conversion_result,
                analysis_result=analysis_result,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            self.error_count += 1
            error_duration = (datetime.now() - processing_start).total_seconds()
            
            logger.error(f"Processing failed after {error_duration:.2f}s: {str(e)}")
            
            return ProcessingResponse.error_response(
                error_message=f"System processing error: {str(e)}",
                error_context={
                    "processing_duration": error_duration,
                    "error_type": type(e).__name__,
                    "processing_phase": "unknown"
                }
            )
    
    async def _execute_conversion_pipeline(self, request: ProcessingRequest) -> ProcessingResult:
        """Isolated conversion processing with resource management"""
        
        # Create mock file object for processing
        class ProcessingFile:
            def __init__(self, content: bytes, metadata: JSONDict):
                self.content = content
                self.name = metadata.get('filename', 'uploaded_file')
                self.size = len(content)
            
            def read(self) -> bytes:
                return self.content
        
        processing_file = ProcessingFile(request.file_content, request.file_metadata)
        
        # Execute file processing
        file_result = await self.file_handler.process_upload(
            processing_file,
            metadata_override=request.file_metadata
        )
        if not file_result.success:
            return file_result
        
        # Execute document conversion
        conversion_result = await self.conversion_engine.convert_stream(
            request.file_content, request.file_metadata
        )
        
        return conversion_result
    
    async def _execute_analysis_pipeline(
        self, 
        request: ProcessingRequest, 
        conversion_result: ProcessingResult
    ) -> Optional[Any]:
        """Isolated AI analysis processing with graceful degradation"""
        
        try:
            # Initialize or retrieve Gemini engine
            gemini_config = GeminiConfig(api_key=request.gemini_api_key)
            engine_id = await self.gemini_manager.create_engine(
                request.gemini_api_key, gemini_config
            )
            
            engine = self.gemini_manager.get_engine(engine_id)
            if not engine:
                logger.warning("Gemini engine creation failed - proceeding without analysis")
                return None
            
            # Execute analysis
            analysis_request = AnalysisRequest(
                content=conversion_result.content,
                analysis_type=AnalysisType(request.analysis_type),
                model=GeminiModel.from_str(request.model_preference)
            )
            
            analysis_result = await engine.analyze_content(analysis_request)
            
            if analysis_result.success:
                logger.info(f"AI analysis completed - Type: {request.analysis_type}")
                return analysis_result
            else:
                logger.warning(f"AI analysis failed: {analysis_result.error_message}")
                return None
                
        except Exception as e:
            logger.warning(f"AI analysis pipeline error (non-fatal): {str(e)}")
            return None
    
    def get_processing_status(self) -> JSONDict:
        """Operational visibility interface"""
        
        success_rate = (
            ((self.processing_count - self.error_count) / self.processing_count * 100)
            if self.processing_count > 0 else 0
        )
        
        average_processing_time = (
            self.total_processing_time / self.processing_count
            if self.processing_count > 0 else 0
        )
        
        return {
            'total_documents_processed': self.processing_count,
            'success_rate_percent': success_rate,
            'error_count': self.error_count,
            'average_processing_time_seconds': average_processing_time,
            'total_processing_time_seconds': self.total_processing_time,
            'status': 'healthy' if success_rate > 90 else 'degraded' if success_rate > 70 else 'unhealthy'
        }


# ==================== UI PRESENTATION LAYER ====================

class GradioResponseFactory:
    """
    Strategic UI generation - separates presentation logic from business logic
    
    Design Principles:  
    - Presentation Separation: UI generation isolated from business logic
    - Consistent Interface: Standardized response patterns
    - Error Communication: Clear, actionable user messaging
    - Progressive Enhancement: Graceful degradation for failed components
    """
    
    def __init__(self, viz_engine: InteractiveVisualizationEngine):
        self.viz_engine = viz_engine
    
    def create_success_response(
        self,
        response: ProcessingResponse
    ) -> Tuple[str, str, str, JSONDict]:
        """Generate comprehensive success UI components"""
        
        # Status display with professional formatting
        processing_time = response.conversion_result.processing_time or 0
        content_length = len(response.conversion_result.content)
        
        status_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="margin: 0 0 10px 0;">✅ Processing Completed Successfully</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div>
                    <strong>Processing Time:</strong><br/>
                    <span style="font-size: 1.2em;">{processing_time:.2f} seconds</span>
                </div>
                <div>
                    <strong>Content Generated:</strong><br/>
                    <span style="font-size: 1.2em;">{content_length:,} characters</span>
                </div>
                <div>
                    <strong>Quality Score:</strong><br/>
                    <span style="font-size: 1.2em;">{response.quality_metrics.get('composite_score', 0):.1f}/10</span>
                </div>
            </div>
        </div>
        """
        
        # Document preview with metadata
        original_preview = self._generate_document_preview(response.conversion_result.metadata)
        
        # Markdown output
        markdown_content = response.conversion_result.content
        
        # Metrics summary for quick review
        quick_metrics = self._extract_summary_metrics(response)
        
        return (
            status_html,
            original_preview,
            markdown_content,
            quick_metrics
        )
    
    def create_error_response(
        self,
        error_message: str,
        error_context: Optional[JSONDict] = None
    ) -> Tuple[str, str, str, JSONDict]:
        """Generate comprehensive error UI components with actionable guidance"""
        
        # Determine error severity and user guidance
        error_type = error_context.get('error_type', 'Unknown') if error_context else 'Unknown'
        processing_phase = error_context.get('processing_phase', 'unknown') if error_context else 'unknown'
        
        # Generate user-friendly error messaging
        if 'Gemini' in error_message or 'API' in error_message:
            user_guidance = "This appears to be an AI analysis issue. The document conversion may have succeeded. Check your API key and try again."
        elif 'conversion' in error_message.lower():
            user_guidance = "Document conversion failed. Please verify your file format is supported and try again."
        elif 'resource' in error_message.lower():
            user_guidance = "System resources are currently limited. Try with a smaller file or wait a moment before retrying."
        else:
            user_guidance = "An unexpected error occurred. Please try again or contact support if the problem persists."
        
        error_html = f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="margin: 0 0 10px 0;">❌ Processing Failed</h3>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>Error Details:</strong><br/>
                {error_message}
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>💡 Recommended Action:</strong><br/>
                {user_guidance}
            </div>
            {f'<p><strong>Error Type:</strong> {error_type} | <strong>Phase:</strong> {processing_phase}</p>' if error_context else ''}
        </div>
        """
        
        return (
            error_html,
            "",  # No preview for errors
            "",  # No markdown content for errors
            {"error": error_message, "timestamp": datetime.now().isoformat()}
        )
    
    def _generate_document_preview(self, metadata: JSONDict) -> str:
        """Generate professional document metadata preview"""
        
        original_file = metadata.get('original_file', {})
        
        return f"""
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 10px 0;">
            <h4 style="color: #495057; margin-bottom: 15px;">📄 Document Information</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 8px; font-weight: bold; color: #6c757d;">Filename:</td>
                    <td style="padding: 8px;">{original_file.get('filename', 'Unknown')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 8px; font-weight: bold; color: #6c757d;">File Size:</td>
                    <td style="padding: 8px;">{original_file.get('size', 0) / 1024:.1f} KB</td>
                </tr>
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 8px; font-weight: bold; color: #6c757d;">Format:</td>
                    <td style="padding: 8px;">{original_file.get('extension', 'Unknown').upper()}</td>
                </tr>
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 8px; font-weight: bold; color: #6c757d;">Processing Date:</td>
                    <td style="padding: 8px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
        </div>
        """
    
    def _extract_summary_metrics(self, response: ProcessingResponse) -> JSONDict:
        """Extract key metrics for UI display"""
        
        basic_metrics = response.quality_metrics.get('basic_metrics', {})
        structural_metrics = response.quality_metrics.get('structural_metrics', {})
        
        return {
            'overall_score': response.quality_metrics.get('composite_score', 0),
            'processing_time': response.conversion_result.processing_time,
            'content_statistics': {
                'total_words': basic_metrics.get('total_words', 0),
                'total_lines': basic_metrics.get('total_lines', 0),
                'total_characters': basic_metrics.get('total_characters', 0)
            },
            'structural_elements': {
                'headers': structural_metrics.get('header_count', 0),
                'lists': structural_metrics.get('list_items', 0),
                'tables': structural_metrics.get('table_rows', 0),
                'links': structural_metrics.get('links', 0)
            },
            'ai_analysis_available': response.analysis_result is not None and response.analysis_result.success if response.analysis_result else False
        }


# ==================== MAIN APPLICATION ASSEMBLY ====================

class MarkItDownTestingApp:
    """
    Strategic application orchestration - human-scale complexity management
    
    Core Design Philosophy:
    - Dependency Injection: All components provided at construction
    - Single Responsibility: UI orchestration only
    - Error Boundaries: Comprehensive error handling at interaction level
    - State Management: Immutable state patterns with clear update paths
    
    This class represents the composition root of the application - where all
    dependencies are wired together and the system boundary is established.
    """
    
    def __init__(
        self,
        orchestrator: DocumentProcessingOrchestrator,
        ui_factory: GradioResponseFactory,
        initial_state: Optional[ApplicationState] = None
    ):
        self.orchestrator = orchestrator
        self.ui_factory = ui_factory
        self.app_state = initial_state or ApplicationState(
            session_id=datetime.now().isoformat()
        )
        
        # Application configuration
        self.config = {
            'title': 'MarkItDown Testing Platform',
            'version': '2.0.0-enterprise',
            'max_file_size_mb': 50,
            'supported_formats': ['.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.html', '.htm', '.csv', '.json', '.xml']
        }
    
    def create_interface(self) -> gr.Blocks:
        """
        Gradio interface assembly with modular component design
        
        Strategic Approach:
        - Component Isolation: Each UI section is self-contained
        - Event Handling: Clean separation between UI events and business logic
        - State Management: Immutable state updates with clear data flow
        - Error Handling: User-friendly error presentation with recovery guidance
        """
        
        with gr.Blocks(
            title=self.config['title'],
            theme=gr.themes.Soft(),
            analytics_enabled=False
        ) as interface:
            
            # Application state for Gradio
            gr_state = gr.State(self.app_state)
            
            # Main header
            self._create_application_header()
            
            # Primary interface tabs
            with gr.Tabs():
                
                # Document Processing Tab
                with gr.TabItem("📁 Document Processing"):
                    processing_components = self._create_processing_interface(gr_state)
                
                # Analytics Dashboard Tab  
                with gr.TabItem("📊 Analysis Dashboard"):
                    analytics_components = self._create_analytics_interface(gr_state)
                
                # System Status Tab
                with gr.TabItem("⚙️ System Status"):
                    self._create_status_interface()
            
            # Wire event handlers with clean separation
            self._wire_event_handlers(processing_components, analytics_components, gr_state)
            
            # Application footer
            self._create_application_footer()
        
        return interface
    
    def _create_application_header(self) -> None:
        """Professional application header with branding"""
        
        gr.HTML(f"""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5em;">🚀 {self.config['title']}</h1>
            <p style="margin: 10px 0; font-size: 1.2em;">Enterprise-Grade Document Conversion Testing with AI-Powered Analysis</p>
            <p style="margin: 0; opacity: 0.9;">
                <em>Version {self.config['version']} | Powered by Microsoft MarkItDown & Google Gemini</em>
            </p>
        </div>
        """)
    
    def _create_processing_interface(self, gr_state: gr.State) -> Dict[str, Any]:
        """Document processing interface with professional UX"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Document Upload & Configuration")
                
                # File upload
                file_upload = gr.File(
                    label="Select Document",
                    file_types=self.config['supported_formats'],
                    type="binary"
                )
                
                # Processing configuration
                with gr.Accordion("🔧 Processing Configuration", open=True):
                    gemini_api_key = gr.Textbox(
                        label="Gemini API Key (Optional)",
                        type="password",
                        placeholder="Enter your Google Gemini API key for AI analysis...",
                        info="Leave empty for basic conversion only"
                    )
                    
                    analysis_type = gr.Dropdown(
                        choices=[
                            ("Quality Analysis", "quality_analysis"),
                            ("Structure Review", "structure_review"),
                            ("Content Summary", "content_summary"),
                            ("Extraction Quality", "extraction_quality")
                        ],
                        value="quality_analysis",
                        label="Analysis Type"
                    )
                    
                    model_preference = gr.Dropdown(
                        choices=[
                            ("Gemini 2.0 Pro (Advanced Reasoning)", GeminiModel.PRO.value),
                            ("Gemini 2.0 Flash (Fast Inference)", GeminiModel.FLASH.value),
                            ("Gemini 2.5 Flash (Enhanced Quality)", GeminiModel.FLASH_25.value),
                            ("Gemini 1.5 Pro (Legacy)", GeminiModel.LEGACY_PRO.value),
                            ("Gemini 1.5 Flash (Legacy)", GeminiModel.LEGACY_FLASH.value)
                        ],
                        value=GeminiModel.PRO.value,
                        label="AI Model Preference"
                    )
                
                # Action buttons
                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Process Document",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🔄 Clear Session",
                        variant="secondary"
                    )
            
            with gr.Column(scale=2):
                # Results display area
                gr.Markdown("### 📊 Processing Results")
                
                status_display = gr.HTML()
                
                with gr.Tabs():
                    with gr.TabItem("📄 Original Document"):
                        original_preview = gr.HTML()
                    
                    with gr.TabItem("📝 Markdown Output"):
                        markdown_output = gr.Code(
                            language="markdown",
                            show_label=False,
                            interactive=False
                        )
                    
                    with gr.TabItem("📈 Quick Metrics"):
                        quick_metrics = gr.JSON()
        
        return {
            'file_upload': file_upload,
            'gemini_api_key': gemini_api_key,
            'analysis_type': analysis_type,
            'model_preference': model_preference,
            'process_btn': process_btn,
            'clear_btn': clear_btn,
            'status_display': status_display,
            'original_preview': original_preview,
            'markdown_output': markdown_output,
            'quick_metrics': quick_metrics
        }
    
    def _create_analytics_interface(self, gr_state: gr.State) -> Dict[str, Any]:
        """Analytics dashboard interface"""
        
        gr.Markdown("### 📊 Document Analysis Dashboard")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
        
        with gr.Row():
            quality_dashboard = gr.Plot(label="Quality Analysis Dashboard")
        
        with gr.Row():
            with gr.Column():
                analysis_summary = gr.Markdown("*Process a document to see analysis results*")
            with gr.Column():
                structure_metrics = gr.JSON(label="Structure Analysis")
        
        return {
            'refresh_btn': refresh_btn,
            'quality_dashboard': quality_dashboard,
            'analysis_summary': analysis_summary,
            'structure_metrics': structure_metrics
        }
    
    def _create_status_interface(self) -> None:
        """System status and health monitoring interface"""
        
        gr.Markdown("### ⚙️ System Status & Health")
        
        with gr.Row():
            with gr.Column():
                system_health = gr.JSON(
                    label="System Health Metrics",
                    value=self._get_system_status()
                )
            
            with gr.Column():
                processing_stats = gr.JSON(
                    label="Processing Statistics",
                    value=self.orchestrator.get_processing_status()
                )
    
    def _create_application_footer(self) -> None:
        """Professional application footer"""
        
        gr.HTML("""
        <div style="text-align: center; padding: 1rem; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 2rem;">
            <p>Built with enterprise-grade architecture principles | 
            <a href="https://github.com/microsoft/markitdown">Microsoft MarkItDown</a> | 
            <a href="https://ai.google.dev/">Google Gemini</a></p>
        </div>
        """)
    
    def _wire_event_handlers(
        self, 
        processing_components: Dict[str, Any], 
        analytics_components: Dict[str, Any],
        gr_state: gr.State
    ) -> None:
        """Wire event handlers with clean separation of concerns"""
        
        # Document processing handler
        processing_components['process_btn'].click(
            fn=self._handle_document_processing,
            inputs=[
                processing_components['file_upload'],
                processing_components['gemini_api_key'],
                processing_components['analysis_type'],
                processing_components['model_preference'],
                gr_state
            ],
            outputs=[
                processing_components['status_display'],
                processing_components['original_preview'],
                processing_components['markdown_output'],
                processing_components['quick_metrics'],
                gr_state
            ],
            show_progress="full"
        )
        
        # Clear session handler
        processing_components['clear_btn'].click(
            fn=self._handle_session_clear,
            inputs=[gr_state],
            outputs=[
                processing_components['status_display'],
                processing_components['original_preview'],
                processing_components['markdown_output'],
                processing_components['quick_metrics'],
                gr_state
            ]
        )
        
        # Analytics refresh handler
        analytics_components['refresh_btn'].click(
            fn=self._handle_analytics_refresh,
            inputs=[gr_state],
            outputs=[
                analytics_components['quality_dashboard'],
                analytics_components['analysis_summary'],
                analytics_components['structure_metrics']
            ]
        )
    
    async def _handle_document_processing(
        self,
        file_obj,
        gemini_api_key: str,
        analysis_type: str,
        model_preference: str,
        current_state: ApplicationState
    ) -> Tuple[str, str, str, JSONDict, ApplicationState]:
        """
        Clean event handler - delegates to orchestrator
        
        Strategic Design:
        - Input Validation: Comprehensive request validation
        - Business Logic Delegation: All processing logic in orchestrator
        - Error Handling: User-friendly error presentation
        - State Management: Immutable state updates
        """
        
        # Input validation
        if not file_obj:
            error_response = self.ui_factory.create_error_response(
                "No file uploaded. Please select a document to process."
            )
            return (*error_response, current_state)
        
        try:
            # Extract file content and metadata
            file_content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            
            file_metadata = {
                'filename': getattr(file_obj, 'name', 'uploaded_file'),
                'size': len(file_content),
                'extension': Path(getattr(file_obj, 'name', 'file.txt')).suffix.lower(),
                'upload_timestamp': datetime.now().isoformat()
            }
            
            # Create processing request
            processing_request = ProcessingRequest(
                file_content=file_content,
                file_metadata=file_metadata,
                gemini_api_key=gemini_api_key.strip() if gemini_api_key else None,
                analysis_type=analysis_type,
                model_preference=model_preference,
                session_context={'session_id': current_state.session_id}
            )
            
            # Execute processing through orchestrator
            processing_response = await self.orchestrator.process_document(processing_request)
            
            # Update application state
            updated_state = current_state.add_processing_result(processing_response)
            
            # Generate UI response
            if processing_response.success:
                ui_response = self.ui_factory.create_success_response(processing_response)
            else:
                ui_response = self.ui_factory.create_error_response(
                    processing_response.error_details,
                    processing_response.processing_metadata
                )
            
            return (*ui_response, updated_state)
            
        except Exception as e:
            logger.error(f"Event handler error: {str(e)}")
            error_response = self.ui_factory.create_error_response(
                f"System error during processing: {str(e)}"
            )
            return (*error_response, current_state)
    
    def _handle_session_clear(
        self, 
        current_state: ApplicationState
    ) -> Tuple[str, str, str, JSONDict, ApplicationState]:
        """Clear session with clean state reset"""
        
        # Create fresh application state
        fresh_state = ApplicationState(
            session_id=datetime.now().isoformat()
        )
        
        # Clear UI components
        clear_html = """
        <div style="background: #e3f2fd; border: 1px solid #2196f3; color: #1976d2; 
                    padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin: 0;">🔄 Session Cleared</h4>
            <p style="margin: 5px 0 0 0;">Ready for new document processing.</p>
        </div>
        """
        
        return (
            clear_html,
            "",  # Clear preview
            "",  # Clear markdown
            {},  # Clear metrics
            fresh_state
        )
    
    def _handle_analytics_refresh(
        self,
        current_state: ApplicationState
    ) -> Tuple[Any, str, JSONDict]:
        """Refresh analytics dashboard with latest data"""
        
        if not current_state.processing_history:
            # Empty state visualization
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="Process documents to see analytics",
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            empty_fig.update_layout(
                title="Analytics Dashboard",
                height=400
            )
            
            return (
                empty_fig,
                "*Process documents to see detailed analysis*",
                {}
            )
        
        # Get latest successful processing result
        latest_result = None
        for result in reversed(current_state.processing_history):
            if result.success:
                latest_result = result
                break
        
        if not latest_result:
            return (
                empty_fig,
                "*No successful processing results available*",
                {}
            )
        
        try:
            # Generate dashboard visualization
            quality_dashboard = self.ui_factory.viz_engine.create_quality_dashboard(
                latest_result.conversion_result,
                latest_result.analysis_result
            )
            
            # Generate analysis summary
            if latest_result.analysis_result:
                analysis_summary = self._format_analysis_summary(latest_result.analysis_result)
            else:
                analysis_summary = "**Basic conversion completed.** Add Gemini API key for AI-powered analysis."
            
            # Generate structure metrics
            structure_metrics = latest_result.quality_metrics.get('structural_metrics', {})
            
            return (
                quality_dashboard,
                analysis_summary,
                structure_metrics
            )
            
        except Exception as e:
            logger.error(f"Analytics refresh error: {str(e)}")
            return (
                empty_fig,
                f"*Analytics refresh failed: {str(e)}*",
                {"error": str(e)}
            )
    
    def _format_analysis_summary(self, analysis_result) -> str:
        """Format AI analysis results for user presentation"""
        
        if not analysis_result or not analysis_result.success:
            return "*AI analysis not available*"
        
        content = analysis_result.content
        analysis_type = analysis_result.analysis_type.value.replace('_', ' ').title()
        
        summary = f"## 🤖 {analysis_type}\n\n"
        summary += f"**Model:** {analysis_result.model_used.value}  \n"
        summary += f"**Processing Time:** {analysis_result.processing_time:.2f}s\n\n"
        
        # Extract key insights based on analysis type
        if 'overall_score' in content:
            summary += f"### 📊 Quality Assessment\n"
            summary += f"**Overall Score:** {content.get('overall_score', 0)}/10\n\n"
            
            scores = []
            if 'structure_score' in content:
                scores.append(f"Structure: {content['structure_score']}/10")
            if 'completeness_score' in content:
                scores.append(f"Completeness: {content['completeness_score']}/10")
            if 'accuracy_score' in content:
                scores.append(f"Accuracy: {content['accuracy_score']}/10")
            
            if scores:
                summary += "**Detailed Scores:** " + " | ".join(scores) + "\n\n"
        
        if 'executive_summary' in content:
            summary += f"### 📋 Executive Summary\n{content['executive_summary']}\n\n"
        
        if 'detailed_feedback' in content:
            feedback = content['detailed_feedback'][:300]
            summary += f"### 💡 Key Insights\n{feedback}{'...' if len(content['detailed_feedback']) > 300 else ''}\n\n"
        
        if 'recommendations' in content and content['recommendations']:
            summary += f"### 🎯 Recommendations\n"
            for i, rec in enumerate(content['recommendations'][:3], 1):
                summary += f"{i}. {rec}\n"
        
        return summary
    
    def _get_system_status(self) -> JSONDict:
        """Get comprehensive system status information"""
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'system': {
                    'status': 'Operational',
                    'cpu_usage_percent': cpu_percent,
                    'memory_usage_percent': memory.percent,
                    'available_memory_gb': round(memory.available / (1024**3), 2),
                    'platform': os.name
                },
                'application': {
                    'version': self.config['version'],
                    'max_file_size_mb': self.config['max_file_size_mb'],
                    'supported_formats': len(self.config['supported_formats']),
                    'session_id': self.app_state.session_id
                },
                'processing': self.orchestrator.get_processing_status()
            }
        except Exception as e:
            return {
                'system': {'status': 'Unknown', 'error': str(e)},
                'application': {'version': self.config['version']},
                'processing': {'status': 'Unknown'}
            }


# ==================== APPLICATION FACTORY & COMPOSITION ROOT ====================

class ApplicationFactory:
    """
    Strategic application composition - dependency injection container
    
    Design Principles:
    - Composition Root: Single location for all dependency wiring
    - Environment Awareness: Different configurations for different environments
    - Component Lifecycle: Proper initialization order and cleanup
    - Configuration Management: Centralized configuration with validation
    """
    
    @staticmethod
    def create_hf_spaces_app() -> MarkItDownTestingApp:
        """
        Factory method for HF Spaces optimized application
        
        Optimizations:
        - Resource Management: Configured for 16GB memory limit
        - Processing Timeouts: Appropriate for shared infrastructure
        - Error Recovery: Graceful degradation under resource pressure
        - Logging Configuration: Production-appropriate logging levels
        """
        
        logger.info("Initializing MarkItDown Testing Platform for HF Spaces deployment")
        
        # Core configuration
        processing_config = ProcessingConfig(
            max_file_size_mb=50,
            max_memory_usage_gb=12.0,
            processing_timeout=300,
            max_concurrent_processes=2
        )
        
        # Resource management
        resource_manager = ResourceManager(processing_config)
        
        # Document processing components
        file_handler = StreamlineFileHandler(resource_manager)
        conversion_engine = HFConversionEngine(resource_manager, processing_config)
        
        # AI analysis components
        gemini_manager = GeminiConnectionManager()
        
        # Analytics and visualization
        viz_config = VisualizationConfig(
            theme=VisualizationConfig.VisualizationTheme.CORPORATE,
            width=800,
            height=600
        )
        viz_engine = InteractiveVisualizationEngine(viz_config)
        quality_calculator = QualityMetricsCalculator()
        
        # Core orchestrator
        orchestrator = DocumentProcessingOrchestrator(
            file_handler=file_handler,
            conversion_engine=conversion_engine,
            gemini_manager=gemini_manager,
            viz_engine=viz_engine,
            quality_calculator=quality_calculator
        )
        
        # UI presentation layer
        ui_factory = GradioResponseFactory(viz_engine)
        
        # Application assembly
        app = MarkItDownTestingApp(
            orchestrator=orchestrator,
            ui_factory=ui_factory
        )
        
        logger.info("Application initialized successfully - Ready for HF Spaces deployment")
        return app
    
    @staticmethod
    def create_local_development_app() -> MarkItDownTestingApp:
        """Factory method for local development with enhanced debugging"""
        
        # Enhanced configuration for local development
        processing_config = ProcessingConfig(
            max_file_size_mb=100,
            max_memory_usage_gb=32.0,
            processing_timeout=600,
            max_concurrent_processes=4
        )
        
        # Enable debug logging for development
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Use same component assembly pattern as HF Spaces
        return ApplicationFactory.create_hf_spaces_app()


# ==================== ENVIRONMENT SETUP & CONFIGURATION ====================

def setup_production_environment() -> None:
    """Configure production environment for optimal performance"""
    
    # Environment variables for HF Spaces
    os.environ.setdefault('GRADIO_TEMP_DIR', '/tmp')
    os.environ.setdefault('HF_HOME', '/tmp')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # System resource verification
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Production environment initialized - Available memory: {memory.available / (1024**3):.2f} GB")
        
        if memory.available < 2 * (1024**3):  # Less than 2GB available
            logger.warning("Low memory detected - enabling aggressive cleanup policies")
            
    except ImportError:
        logger.warning("psutil not available - resource monitoring disabled")


def create_gradio_app() -> gr.Blocks:
    """
    Main application factory for Gradio deployment
    
    This is the primary entry point for the application, designed to be called
    by Gradio's deployment infrastructure.
    """
    
    setup_production_environment()
    
    # Create application instance
    app = ApplicationFactory.create_hf_spaces_app()
    
    # Create Gradio interface
    interface = app.create_interface()
    
    return interface


# ==================== MAIN ENTRY POINT ====================

def main():
    """
    Main application entry point for direct execution
    
    Supports both development and production deployment modes with
    appropriate configuration for each environment.
    """
    
    setup_production_environment()
    
    # Create and configure application
    app = ApplicationFactory.create_hf_spaces_app()
    interface = app.create_interface()
    
    # Launch configuration optimized for HF Spaces
    launch_kwargs = {
        'server_name': '0.0.0.0',
        'server_port': int(os.environ.get('PORT', 7860)),
        'share': False,  # HF Spaces handles sharing
        'show_error': True,
        'max_file_size': f"{50 * 1024 * 1024}b",  # 50MB limit
        'allowed_paths': ['/tmp'],
        'root_path': os.environ.get('GRADIO_ROOT_PATH', '')
    }
    
    # Launch application
    try:
        logger.info(f"Launching MarkItDown Testing Platform on port {launch_kwargs['server_port']}")
        interface.launch(**launch_kwargs)
    except Exception as e:
        logger.error(f"Application launch failed: {str(e)}")
        raise


# ==================== MODULE INTERFACE ====================

# Public API for external integration
__all__ = [
    'MarkItDownTestingApp',
    'ApplicationFactory', 
    'ProcessingRequest',
    'ProcessingResponse',
    'create_gradio_app',
    'main'
]


if __name__ == "__main__":
    main()
