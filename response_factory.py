"""
Strategic Response Processing Architecture - UI Response Abstraction

Design Philosophy:
"Transform business results into compelling user experiences"

This module provides a clean separation between business logic outcomes and 
UI presentation, enabling consistent user experiences and simplified testing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

from service_layer import ServiceResult, SessionContext
from app_logic import ProcessingResponse
from pydantic import JsonValue

logger = logging.getLogger(__name__)

JSONDict = Dict[str, JsonValue]


# ==================== RESPONSE ABSTRACTIONS ====================

@dataclass(frozen=True)
class UIResponse:
    """Standardized UI response structure"""
    
    status_html: str
    original_preview: str 
    content_display: str
    metrics_data: JSONDict
    
    @classmethod
    def success(
        cls,
        status_message: str,
        content: str,
        preview: str = "",
        metrics: JSONDict = None
    ) -> "UIResponse":
        """Create successful UI response"""
        
        return cls(
            status_html=cls._format_success_status(status_message),
            original_preview=preview,
            content_display=content,
            metrics_data=metrics or {}
        )
    
    @classmethod
    def error(
        cls,
        error_message: str,
        error_context: Optional[str] = None,
        guidance: Optional[str] = None
    ) -> "UIResponse":
        """Create error UI response"""
        
        return cls(
            status_html=cls._format_error_status(error_message, error_context, guidance),
            original_preview="",
            content_display="",
            metrics_data={"error": error_message, "timestamp": datetime.now().isoformat()}
        )
    
    @staticmethod
    def _format_success_status(message: str) -> str:
        """Format success status with consistent styling"""
        
        return f"""
        <div style=\"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;\">
            <h3 style=\"margin: 0 0 10px 0;\">✅ {message}</h3>
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>🎉 Processing completed successfully!</strong><br/>
                <em>Results are displayed in the tabs below.</em>
            </div>
        </div>
        """
    
    @staticmethod
    def _format_error_status(message: str, context: Optional[str], guidance: Optional[str]) -> str:
        """Format error status with helpful guidance"""
        
        error_html = f"""
        <div style=\"background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;\">
            <h3 style=\"margin: 0 0 10px 0;\">❌ Processing Failed</h3>
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>Error Details:</strong><br/>
                {message}
            </div>
        """
        
        if context:
            error_html += f"""
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>Context:</strong><br/>
                {context}
            </div>
            """
        
        if guidance:
            error_html += f"""
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>💡 Recommended Action:</strong><br/>
                {guidance}
            </div>
            """
        
        error_html += "</div>"
        return error_html


@dataclass(frozen=True)
class AnalyticsResponse:
    """Structured response for analytics operations"""
    
    dashboard_figure: Optional[Any]
    summary_markdown: str
    metrics_json: JSONDict
    
    @classmethod
    def success(
        cls,
        figure: Any,
        summary: str,
        metrics: JSONDict
    ) -> "AnalyticsResponse":
        """Create successful analytics response"""
        
        return cls(
            dashboard_figure=figure,
            summary_markdown=summary,
            metrics_json=metrics
        )
    
    @classmethod
    def empty(cls, message: str = "No data available for analytics") -> "AnalyticsResponse":
        """Create empty analytics response"""
        
        import plotly.graph_objects as go
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        empty_fig.update_layout(title="Analytics Dashboard", height=400)
        
        return cls(
            dashboard_figure=empty_fig,
            summary_markdown=f"*{message}*",
            metrics_json={}
        )


# ==================== RESPONSE PROCESSORS ====================

class ProcessingResultProcessor:
    """Specialized processor for document processing results"""
    
    def __init__(self):
        self.processing_count = 0
        
    def process_success_result(self, service_result: ServiceResult) -> UIResponse:
        """Process successful document processing result"""
        
        self.processing_count += 1
        
        if not service_result.success or not service_result.data:
            return UIResponse.error("Invalid processing result received")
        
        processing_response: ProcessingResponse = service_result.data
        processing_time = service_result.metadata.get('processing_time', 0)

        conversion_result = processing_response.conversion_result
        conversion_content = getattr(conversion_result, 'content', '') if conversion_result else ''
        # Defensive: mocks may fabricate attributes; coerce to safe types
        if not isinstance(conversion_content, (str, bytes, bytearray)):
            try:
                conversion_content = str(conversion_content)
            except Exception:
                conversion_content = ''
        conversion_metadata = getattr(conversion_result, 'metadata', {}) if conversion_result else {}
        quality_metrics = (
            processing_response.quality_metrics
            if isinstance(processing_response.quality_metrics, Mapping)
            else {}
        )

        # Extract core metrics
        try:
            if isinstance(conversion_content, (bytes, bytearray)):
                content_length = len(conversion_content)
            elif isinstance(conversion_content, str):
                content_length = len(conversion_content)
            else:
                content_length = len(str(conversion_content)) if conversion_content else 0
        except Exception:
            content_length = 0
        quality_score = quality_metrics.get('composite_score', 0)
        
        # Build status message
        status_message = f"Document Processing Completed (#{self.processing_count})"
        
        # Generate document preview
        original_preview = self._generate_document_preview(conversion_metadata)
        
        # Process content based on analysis type
        content_display = self._process_content_display(processing_response)
        
        # Extract metrics
        metrics = self._extract_processing_metrics(processing_response, processing_time)
        
        return UIResponse.success(
            status_message=status_message,
            content=content_display,
            preview=original_preview,
            metrics=metrics
        )
    
    def process_error_result(self, service_result: ServiceResult) -> UIResponse:
        """Process failed document processing result"""
        
        error_message = service_result.error_message or "Unknown processing error"
        processing_time = service_result.metadata.get('processing_time', 0)
        
        # Determine error guidance based on error type
        guidance = self._generate_error_guidance(error_message)
        
        context = f"Processing failed after {processing_time:.1f} seconds"
        if service_result.metadata.get('exception_type'):
            context += f" ({service_result.metadata['exception_type']})"
        
        return UIResponse.error(
            error_message=error_message,
            error_context=context,
            guidance=guidance
        )
    
    def _process_content_display(self, response: ProcessingResponse) -> str:
        """Process content for display based on analysis results"""
        
        # If AI analysis is available and successful, format that
        if response.analysis_result and getattr(response.analysis_result, 'success', False):
            return self._format_ai_analysis_content(response.analysis_result)
        
        # Otherwise, return basic conversion content
        if response.conversion_result and getattr(response.conversion_result, 'content', None):
            content = response.conversion_result.content
            if isinstance(content, (bytes, bytearray)):
                try:
                    return content.decode('utf-8', errors='ignore')
                except Exception:
                    return str(content)
            if not isinstance(content, str):
                return str(content)
            return content
        
        return "No content available"
    
    def _format_ai_analysis_content(self, analysis_result) -> str:
        """Format AI analysis results for display"""
        
        analysis_type = getattr(analysis_result, 'analysis_type', 'generic')
        analysis_type = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
        ai_content = self._ensure_json_serializable(
            getattr(analysis_result, 'content', {})
        )

        if not isinstance(ai_content, Mapping):
            ai_content = {'analysis_content': ai_content}

        if analysis_type == "quality_analysis":
            return self._format_quality_analysis(ai_content)
        elif analysis_type == "structure_review":
            return self._format_structure_analysis(ai_content)
        elif analysis_type == "content_summary":
            return self._format_content_summary(ai_content)
        elif analysis_type == "extraction_quality":
            return self._format_extraction_analysis(ai_content)
        else:
            return self._format_generic_analysis(ai_content)
    
    def _format_quality_analysis(self, content: Dict) -> str:
        """Format quality analysis for UI display"""
        
        markdown = f"""# 📊 Quality Analysis Results

## Overall Assessment
**Quality Score**: {content.get('overall_score', 'N/A')}/10

## Detailed Metrics
- **Structure Score**: {content.get('structure_score', 'N/A')}/10 - Збереження заголовків, списків, таблиць
- **Completeness Score**: {content.get('completeness_score', 'N/A')}/10 - Повнота інформації з оригіналу  
- **Accuracy Score**: {content.get('accuracy_score', 'N/A')}/10 - Точність передачі форматування
- **Readability Score**: {content.get('readability_score', 'N/A')}/10 - Оптимізація для AI-споживання

## 🤖 AI Analysis Feedback
{content.get('detailed_feedback', 'No detailed feedback available')}

## 💡 Recommendations
"""
        
        recommendations = content.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
        else:
            markdown += "No specific recommendations available.\n"
        
        return markdown
    
    def _format_structure_analysis(self, content: Dict) -> str:
        """Format structure analysis for UI display"""
        
        markdown = f"""# 🏗️ Document Structure Analysis

## Organization Score
**Structure Quality**: {content.get('organization_score', 'N/A')}/10

## Document Outline
```
{content.get('document_outline', 'No outline available')}
```

## Structural Elements
"""
        
        heading_analysis = content.get('heading_analysis', {})
        if heading_analysis:
            markdown += "### Heading Analysis\n"
            for level, count in heading_analysis.items():
                markdown += f"- **{level}**: {count} occurrences\n"
        
        return markdown
    
    def _format_content_summary(self, content: Dict) -> str:
        """Format content summary for UI display"""
        
        markdown = f"""# 📝 Content Summary & Analysis

## Executive Summary
{content.get('executive_summary', 'No summary available')}

## Content Quality Score
**Information Value**: {content.get('content_quality', 'N/A')}/10

## Main Topics
"""
        
        topics = content.get('main_topics', [])
        if topics:
            for topic in topics:
                markdown += f"- {topic}\n"
        else:
            markdown += "No main topics identified\n"
        
        return markdown
    
    def _format_extraction_analysis(self, content: Dict) -> str:
        """Format extraction quality analysis for UI display"""
        
        markdown = f"""# 🔍 Extraction Quality Assessment

## Overall Extraction Score
**Quality Rating**: {content.get('extraction_score', 'N/A')}/10

## Analysis Confidence
**Confidence Level**: {content.get('confidence_level', 'N/A')}

## Key Findings
{content.get('data_accuracy', 'No accuracy assessment available')}

## Quality Recommendations
"""
        
        recommendations = content.get('quality_recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
        
        return markdown
    
    def _format_generic_analysis(self, content: Dict) -> str:
        """Generic formatter for unknown analysis types"""
        
        import json
        return f"""# 🤖 AI Analysis Results

## Analysis Output
```json
{json.dumps(content, indent=2)}
```

*This analysis type uses a generic formatter. Consider adding specific formatting for better readability.*
"""
    
    def _generate_document_preview(self, metadata: JSONDict) -> str:
        """Generate document metadata preview"""
        
        if not metadata or not isinstance(metadata, Mapping):
            return "<div><em>No document metadata available</em></div>"
        
        file_info = ""
        for key, value in metadata.items():
            if key.startswith('_'):  # Skip private metadata
                continue
            display_key = key.replace('_', ' ').title()
            file_info += f"<li><strong>{display_key}:</strong> {value}</li>"
        
        return f"""
        <div style=\"background: #f4f6ff; border-radius: 10px; padding: 20px;\">
            <h4 style=\"margin-top: 0;\">📄 Document Metadata</h4>
            <ul style=\"list-style: none; padding: 0; margin: 0; line-height: 1.6;\">
                {file_info}
            </ul>
        </div>
        """
    
    def _extract_processing_metrics(self, response: ProcessingResponse, processing_time: float) -> JSONDict:
        """Extract comprehensive metrics from processing response"""
        
        quality_metrics = (
            response.quality_metrics
            if isinstance(response.quality_metrics, Mapping)
            else {}
        )
        basic_metrics = quality_metrics.get("basic_metrics", {})
        structural_metrics = quality_metrics.get("structural_metrics", {})

        if not isinstance(basic_metrics, Mapping):
            basic_metrics = {}
        if not isinstance(structural_metrics, Mapping):
            structural_metrics = {}
        
        return {
            "processing_time": processing_time,
            "content_statistics": {
                "total_words": basic_metrics.get("total_words", 0),
                "total_lines": basic_metrics.get("total_lines", 0),
                "total_characters": basic_metrics.get("total_characters", 0),
            },
            "structural_elements": {
                "headers": structural_metrics.get("header_count", 0),
                "lists": structural_metrics.get("list_items", 0),
                "tables": structural_metrics.get("table_rows", 0),
                "links": structural_metrics.get("links", 0),
            },
            "quality_assessment": {
                "composite_score": quality_metrics.get("composite_score", 0),
                "ai_analysis_available": bool(response.analysis_result and response.analysis_result.success),
            },
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "success": response.success,
            }
        }

    def _ensure_json_serializable(self, value: Any) -> Any:
        """Recursively convert value into JSON-serializable form"""

        if isinstance(value, Mapping):
            return {
                str(key): self._ensure_json_serializable(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._ensure_json_serializable(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)
    
    def _generate_error_guidance(self, error_message: str) -> str:
        """Generate contextual guidance based on error type"""
        
        error_lower = error_message.lower()
        
        if "gemini" in error_lower or "api" in error_lower:
            return (
                "This appears to be an AI analysis issue. The document conversion may have succeeded. "
                "Check your API key and try again, or disable AI analysis to continue with conversion only."
            )
        elif "conversion" in error_lower or "markitdown" in error_lower:
            return (
                "Document conversion failed. Please verify your file format is supported and try again."
            )
        elif "resource" in error_lower or "memory" in error_lower:
            return (
                "System resources are currently limited. Try with a smaller file or wait a moment before retrying."
            )
        elif "timeout" in error_lower:
            return (
                "Processing timed out. Consider disabling AI analysis for faster processing or try with a smaller file."
            )
        else:
            return (
                "An unexpected error occurred. Please try again or contact support if the problem persists."
            )


class AnalyticsResultProcessor:
    """Specialized processor for analytics results"""
    
    def process_analytics_result(self, service_result: ServiceResult) -> AnalyticsResponse:
        """Process analytics service result into UI response"""
        
        if not service_result.success:
            return AnalyticsResponse.empty(
                f"Analytics generation failed: {service_result.error_message}"
            )
        
        if not service_result.data:
            return AnalyticsResponse.empty("No analytics data available")
        
        analytics_data = service_result.data
        
        # Extract components
        dashboard_figure = analytics_data.get('dashboard_figure')
        summary_text = analytics_data.get('summary', '*No summary available*')
        metrics_data = analytics_data.get('metrics', {})
        
        return AnalyticsResponse.success(
            figure=dashboard_figure,
            summary=summary_text,
            metrics=metrics_data
        )


class SessionResultProcessor:
    """Specialized processor for session-related operations"""
    
    def create_clear_session_response(self) -> UIResponse:
        """Create UI response for session clear operation"""
        
        clear_html = """
        <div style=\"background: #e3f2fd; border: 1px solid #2196f3; color: #1976d2; \
                    padding: 15px; border-radius: 8px; margin: 10px 0;\">
            <h4 style=\"margin: 0;\">🔄 Session Cleared</h4>
            <p style=\"margin: 5px 0 0 0;\">Ready for new document processing.</p>
        </div>
        """
        
        return UIResponse(
            status_html=clear_html,
            original_preview="",
            content_display="",
            metrics_data={"action": "session_cleared", "timestamp": datetime.now().isoformat()}
        )


# ==================== UNIFIED RESPONSE FACTORY ====================

class StrategicResponseFactory:
    """
    Central factory for all UI response generation
    
    Design Philosophy: Single source of truth for UI response formatting
    """
    
    def __init__(self):
        self.processing_processor = ProcessingResultProcessor()
        self.analytics_processor = AnalyticsResultProcessor()
        self.session_processor = SessionResultProcessor()
        
        # Track factory usage
        self.response_count = 0
        self.response_types = {}
        
    def create_processing_response(self, service_result: ServiceResult) -> Tuple[str, str, str, JSONDict]:
        """Create processing response tuple for Gradio interface"""
        
        self.response_count += 1
        self._track_response_type('processing')
        
        if service_result.success:
            ui_response = self.processing_processor.process_success_result(service_result)
        else:
            ui_response = self.processing_processor.process_error_result(service_result)
        
        logger.debug(f"Processing response created (#{self.response_count})")
        
        return (
            ui_response.status_html,
            ui_response.original_preview,
            ui_response.content_display,
            ui_response.metrics_data
        )
    
    def create_analytics_response(self, service_result: ServiceResult) -> Tuple[Any, str, JSONDict]:
        """Create analytics response tuple for Gradio interface"""
        
        self.response_count += 1
        self._track_response_type('analytics')
        
        analytics_response = self.analytics_processor.process_analytics_result(service_result)
        
        logger.debug(f"Analytics response created (#{self.response_count})")
        
        return (
            analytics_response.dashboard_figure,
            analytics_response.summary_markdown,
            analytics_response.metrics_json
        )
    
    def create_session_clear_response(self) -> Tuple[str, str, str, JSONDict]:
        """Create session clear response tuple for Gradio interface"""
        
        self.response_count += 1
        self._track_response_type('session_clear')
        
        ui_response = self.session_processor.create_clear_session_response()
        
        logger.debug(f"Session clear response created (#{self.response_count})")
        
        return (
            ui_response.status_html,
            ui_response.original_preview,
            ui_response.content_display,
            ui_response.metrics_data
        )
    
    def create_validation_response(self, validation_result: ServiceResult) -> str:
        """Create validation status message"""
        
        self.response_count += 1
        self._track_response_type('validation')
        
        if validation_result.success:
            return "✅ **Configuration validated successfully.**"
        else:
            return f"⚠️ **Configuration issue**: {validation_result.error_message}"
    
    def get_factory_metrics(self) -> JSONDict:
        """Get factory usage metrics for monitoring"""
        
        return {
            'total_responses_created': self.response_count,
            'response_type_distribution': self.response_types.copy(),
            'factory_health': 'healthy' if self.response_count > 0 else 'unused'
        }
    
    def _track_response_type(self, response_type: str):
        """Track response type usage for analytics"""
        
        if response_type not in self.response_types:
            self.response_types[response_type] = 0
        self.response_types[response_type] += 1


# ==================== EXPORTS ====================

__all__ = [
    'UIResponse',
    'AnalyticsResponse',
    'ProcessingResultProcessor',
    'AnalyticsResultProcessor',
    'SessionResultProcessor',
    'StrategicResponseFactory',
]
