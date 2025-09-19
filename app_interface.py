"""Gradio interface assembly for the MarkItDown Testing Platform - ВИПРАВЛЕНА ВЕРСІЯ."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from app_logic import (
    DocumentProcessingOrchestrator,
    HFConversionEngine,
    JSONDict,
    ProcessingConfig,
    ProcessingRequest,
    ProcessingResponse,
    QualityMetricsCalculator,
    ResourceManager,
    StreamlineFileHandler,
)
from llm.gemini_connector import GeminiConnectionManager, GeminiModel
from visualization.analytics_engine import (
    InteractiveVisualizationEngine,
    VisualizationConfig,
)


logger = logging.getLogger(__name__)

load_dotenv()
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


if hasattr(gr.Blocks, "get_api_info"):
    def _suppress_api_info(self):
        return {"named_endpoints": {}, "unnamed_endpoints": []}

    gr.Blocks.get_api_info = _suppress_api_info


@dataclass
class ApplicationState:
    """Immutable state container for UI session data."""

    session_id: str
    processing_history: Tuple[ProcessingResponse, ...] = field(default_factory=tuple)
    current_gemini_engine_id: Optional[str] = None
    user_preferences: JSONDict = field(default_factory=dict)
    system_metrics: JSONDict = field(default_factory=dict)

    def add_processing_result(self, response: ProcessingResponse) -> "ApplicationState":
        new_history = self.processing_history + (response,)
        return ApplicationState(
            session_id=self.session_id,
            processing_history=new_history,
            current_gemini_engine_id=self.current_gemini_engine_id,
            user_preferences=self.user_preferences,
            system_metrics=self.system_metrics,
        )


class GradioResponseFactory:
    """Creates UI-ready artifacts from processing results - ВИПРАВЛЕНА ВЕРСІЯ з правильною логікою відображення."""

    def __init__(self, viz_engine: InteractiveVisualizationEngine) -> None:
        self.viz_engine = viz_engine

    def create_success_response(
        self, response: ProcessingResponse
    ) -> Tuple[str, str, str, JSONDict]:
        """
        🚨 ВИПРАВЛЕНА ЛОГІКА: Правильне відображення результатів залежно від analysis_type
        
        Strategic Architecture Decision:
        - Показуємо AI analysis результат якщо доступний та успішний
        - Різні analysis_type режими показують різні форматовані результати
        - Graceful fallback до base conversion якщо AI недоступний
        """
        processing_time = response.conversion_result.processing_time or 0
        content_length = len(response.conversion_result.content)

        status_html = f"""
        <div style=\"background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;\">
            <h3 style=\"margin: 0 0 10px 0;\">✅ Processing Completed Successfully</h3>
            <div style=\"display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;\">
                <div>
                    <strong>Processing Time:</strong><br/>
                    <span style=\"font-size: 1.2em;\">{processing_time:.2f} seconds</span>
                </div>
                <div>
                    <strong>Content Generated:</strong><br/>
                    <span style=\"font-size: 1.2em;\">{content_length:,} characters</span>
                </div>
                <div>
                    <strong>Quality Score:</strong><br/>
                    <span style=\"font-size: 1.2em;\">{response.quality_metrics.get('composite_score', 0):.1f}/10</span>
                </div>
            </div>
        </div>
        """

        original_preview = self._generate_document_preview(response.conversion_result.metadata)
        
        # 🚨 КРИТИЧНЕ ВИПРАВЛЕННЯ: Правильна логіка вибору контенту для відображення
        if response.analysis_result and response.analysis_result.success:
            # Показуємо AI-обробний результат залежно від analysis_type
            analysis_type_value = response.analysis_result.analysis_type.value
            ai_content = response.analysis_result.content
            
            status_html += f"""
            <div style="background: #e8f5e8; border: 1px solid #4caf50; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong>🤖 AI Analysis Active:</strong> {analysis_type_value.replace('_', ' ').title()}<br/>
                <strong>Model Used:</strong> {response.analysis_result.model_used.value}<br/>
                <strong>Processing Time:</strong> {response.analysis_result.processing_time:.2f}s
            </div>
            """
            
            if analysis_type_value == "quality_analysis":
                markdown_content = self._format_quality_analysis(ai_content)
            elif analysis_type_value == "structure_review":
                markdown_content = self._format_structure_analysis(ai_content)
            elif analysis_type_value == "content_summary":
                markdown_content = self._format_content_summary(ai_content)
            elif analysis_type_value == "extraction_quality":
                markdown_content = self._format_extraction_analysis(ai_content)
            else:
                # Fallback до formatted AI result
                markdown_content = self._format_generic_ai_result(ai_content)
                
        else:
            # Fallback до базової конвертації якщо AI недоступний або неуспішний
            markdown_content = response.conversion_result.content
            
            if response.analysis_result and not response.analysis_result.success:
                status_html += f"""
                <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <strong>⚠️ AI Analysis Failed:</strong> {response.analysis_result.error_message}<br/>
                    <strong>Showing Base Conversion</strong>
                </div>
                """
        
        quick_metrics = self._extract_summary_metrics(response)

        return (
            status_html,
            original_preview,
            markdown_content,
            quick_metrics,
        )

    def _format_quality_analysis(self, ai_content: Dict) -> str:
        """Форматує результати Quality Analysis для UI display"""
        
        markdown = f"""# 📊 Quality Analysis Results

## Overall Assessment
**Quality Score**: {ai_content.get('overall_score', 'N/A')}/10

## Detailed Metrics
- **Structure Score**: {ai_content.get('structure_score', 'N/A')}/10 - Збереження заголовків, списків, таблиць
- **Completeness Score**: {ai_content.get('completeness_score', 'N/A')}/10 - Повнота інформації з оригіналу  
- **Accuracy Score**: {ai_content.get('accuracy_score', 'N/A')}/10 - Точність передачі форматування
- **Readability Score**: {ai_content.get('readability_score', 'N/A')}/10 - Оптимізація для AI-споживання

## 🤖 AI Analysis Feedback
{ai_content.get('detailed_feedback', 'No detailed feedback available')}

## 💡 Recommendations
"""
        
        recommendations = ai_content.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
        else:
            markdown += "No specific recommendations available.\n"
        
        # Додаємо detected elements якщо доступні
        detected_elements = ai_content.get('detected_elements', {})
        if detected_elements:
            markdown += f"""
## 🔍 Detected Document Elements
"""
            for element, count in detected_elements.items():
                markdown += f"- **{element.replace('_', ' ').title()}**: {count}\n"
        
        return markdown

    def _format_structure_analysis(self, ai_content: Dict) -> str:
        """Форматує результати Structure Review для UI display"""
        
        markdown = f"""# 🏗️ Document Structure Analysis

## Document Outline
```
{ai_content.get('document_outline', 'No outline available')}
```

## Heading Analysis
"""
        
        heading_analysis = ai_content.get('heading_analysis', {})
        if heading_analysis:
            for level, count in heading_analysis.items():
                markdown += f"- **{level}**: {count} occurrences\n"
        else:
            markdown += "No heading analysis available\n"

        markdown += f"""
## Organization Score
**Structure Quality**: {ai_content.get('organization_score', 'N/A')}/10

## List Analysis
"""
        
        list_analysis = ai_content.get('list_analysis', {})
        if list_analysis:
            markdown += f"- **Total Lists**: {list_analysis.get('total_lists', 0)}\n"
            markdown += f"- **Nested Lists**: {list_analysis.get('nested_lists', 0)}\n"
            markdown += f"- **List Items**: {list_analysis.get('total_items', 0)}\n"
        
        markdown += f"""
## Table Analysis  
"""
        
        table_analysis = ai_content.get('table_analysis', {})
        if table_analysis:
            markdown += f"- **Total Tables**: {table_analysis.get('table_count', 0)}\n"
            markdown += f"- **Table Quality**: {table_analysis.get('formatting_quality', 'N/A')}\n"
        
        markdown += f"""
## 💡 Structure Recommendations
"""
        
        recommendations = ai_content.get('structure_recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
        else:
            markdown += "Document structure is well-organized.\n"
            
        return markdown

    def _format_content_summary(self, ai_content: Dict) -> str:
        """Форматує результати Content Summary для UI display"""
        
        markdown = f"""# 📝 Content Summary & Analysis

## Executive Summary
{ai_content.get('executive_summary', 'No summary available')}

## Main Topics
"""
        
        topics = ai_content.get('main_topics', [])
        if topics:
            for topic in topics:
                markdown += f"- {topic}\n"
        else:
            markdown += "No main topics identified\n"
        
        markdown += f"""
## Document Classification
"""
        
        classification = ai_content.get('document_classification', {})
        if classification:
            markdown += f"- **Type**: {classification.get('type', 'Unknown')}\n"
            markdown += f"- **Purpose**: {classification.get('purpose', 'Unknown')}\n"
            markdown += f"- **Target Audience**: {classification.get('audience', 'Unknown')}\n"

        markdown += f"""
## Content Quality Score
**Information Value**: {ai_content.get('content_quality', 'N/A')}/10

## Key Information
"""
        
        key_info = ai_content.get('key_information', [])
        if key_info:
            for info in key_info:
                markdown += f"- {info}\n"
        else:
            markdown += "No key information extracted\n"
        
        # Додаємо content metrics якщо доступні
        content_metrics = ai_content.get('content_metrics', {})
        if content_metrics:
            markdown += f"""
## Content Metrics
- **Word Count**: {content_metrics.get('word_count', 'N/A')}
- **Complexity Level**: {content_metrics.get('complexity_level', 'N/A')}
"""
            
        return markdown

    def _format_extraction_analysis(self, ai_content: Dict) -> str:
        """Форматує результати Extraction Quality для UI display"""
        
        markdown = f"""# 🔍 Extraction Quality Assessment

## Overall Extraction Score
**Quality Rating**: {ai_content.get('extraction_score', 'N/A')}/10

## Data Accuracy Assessment
{ai_content.get('data_accuracy', 'No accuracy assessment available')}

## Context Preservation
**Meaning Retention**: {ai_content.get('context_preservation', 'No context analysis available')}

## Formatting Quality
**Original Structure**: {ai_content.get('formatting_quality', 'No formatting analysis available')}

## Completeness Indicators
{ai_content.get('completeness_indicators', 'No completeness data available')}

## Conversion Artifacts
"""
        
        artifacts = ai_content.get('conversion_artifacts', [])
        if artifacts:
            for artifact in artifacts:
                markdown += f"- ⚠️ {artifact}\n"
        else:
            markdown += "✅ No conversion artifacts detected\n"

        markdown += f"""
## 💡 Quality Recommendations
"""
        
        recommendations = ai_content.get('quality_recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
        else:
            markdown += "Extraction quality is satisfactory.\n"
        
        # Додаємо confidence level
        confidence = ai_content.get('confidence_level', 'N/A')
        markdown += f"""
## Analysis Confidence
**Confidence Level**: {confidence}
"""
            
        return markdown

    def _format_generic_ai_result(self, ai_content: Dict) -> str:
        """Generic formatter для невідомих analysis types"""
        
        markdown = f"""# 🤖 AI Analysis Results

## Analysis Output
```json
{ai_content}
```

*This analysis type uses a generic formatter. Consider adding specific formatting for better readability.*
"""
        return markdown

    def create_error_response(
        self, error_message: str, error_context: Optional[JSONDict] = None
    ) -> Tuple[str, str, str, JSONDict]:
        error_type = error_context.get("error_type", "Unknown") if error_context else "Unknown"
        processing_phase = error_context.get("processing_phase", "unknown") if error_context else "unknown"

        if "Gemini" in error_message or "API" in error_message:
            user_guidance = (
                "This appears to be an AI analysis issue. The document conversion may have succeeded. "
                "Check your API key and try again."
            )
        elif "conversion" in error_message.lower():
            user_guidance = (
                "Document conversion failed. Please verify your file format is supported and try again."
            )
        elif "resource" in error_message.lower():
            user_guidance = (
                "System resources are currently limited. Try with a smaller file or wait a moment before retrying."
            )
        else:
            user_guidance = (
                "An unexpected error occurred. Please try again or contact support if the problem persists."
            )

        error_html = f"""
        <div style=\"background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;\">
            <h3 style=\"margin: 0 0 10px 0;\">❌ Processing Failed</h3>
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>Error Details:</strong><br/>
                {error_message}
            </div>
            <div style=\"background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;\">
                <strong>💡 Recommended Action:</strong><br/>
                {user_guidance}
            </div>
            {f'<p><strong>Error Type:</strong> {error_type} | <strong>Phase:</strong> {processing_phase}</p>' if error_context else ''}
        </div>
        """

        return (
            error_html,
            "",
            "",
            {"error": error_message, "timestamp": datetime.now().isoformat()},
        )

    def _generate_document_preview(self, metadata: JSONDict) -> str:
        file_info = "".join(
            f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
            for key, value in metadata.items()
        )
        return f"""
        <div style=\"background: #f4f6ff; border-radius: 10px; padding: 20px;\">
            <h4 style=\"margin-top: 0;\">📄 Document Metadata</h4>
            <ul style=\"list-style: none; padding: 0; margin: 0; line-height: 1.6;\">
                {file_info}
            </ul>
        </div>
        """

    def _extract_summary_metrics(self, response: ProcessingResponse) -> JSONDict:
        basic_metrics = response.quality_metrics.get("basic_metrics", {})
        structural_metrics = response.quality_metrics.get("structural_metrics", {})

        return {
            "processing_time": response.conversion_result.processing_time,
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
            "ai_analysis_available": bool(
                response.analysis_result and getattr(response.analysis_result, "success", False)
            ),
        }


class MarkItDownTestingApp:
    """Assembles Gradio interface and wires UI event handlers."""

    def __init__(
        self,
        orchestrator: DocumentProcessingOrchestrator,
        ui_factory: GradioResponseFactory,
        initial_state: Optional[ApplicationState] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.ui_factory = ui_factory
        self.app_state = initial_state or ApplicationState(session_id=datetime.now().isoformat())
        self.default_gemini_key = DEFAULT_GEMINI_API_KEY

        self.config = {
            "title": "MarkItDown Testing Platform",
            "version": "2.0.0-enterprise",
            "max_file_size_mb": 50,
            "supported_formats": [
                ".pdf",
                ".docx",
                ".pptx",
                ".xlsx",
                ".txt",
                ".html",
                ".htm",
                ".csv",
                ".json",
                ".xml",
            ],
        }

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title=self.config["title"],
            theme=gr.themes.Soft(),
            analytics_enabled=False,
        ) as interface:
            gr_state = gr.State(self.app_state)

            self._create_application_header()

            with gr.Tabs():
                with gr.TabItem("📁 Document Processing"):
                    processing_components = self._create_processing_interface(gr_state)

                with gr.TabItem("📊 Analysis Dashboard"):
                    analytics_components = self._create_analytics_interface(gr_state)

                with gr.TabItem("⚙️ System Status"):
                    self._create_status_interface()

            self._wire_event_handlers(processing_components, analytics_components, gr_state)
            self._create_application_footer()

        return interface

    def _create_application_header(self) -> None:
        gr.HTML(
            f"""
        <div style=\"text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); \
                    color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;\">
            <h1 style=\"margin: 0; font-size: 2.5em;\">🚀 {self.config['title']}</h1>
            <p style=\"margin: 10px 0; font-size: 1.2em;\">Enterprise-Grade Document Conversion Testing with AI-Powered Analysis</p>
            <p style=\"margin: 0; opacity: 0.9;\">
                <em>Version {self.config['version']} | Powered by Microsoft MarkItDown & Google Gemini</em>
            </p>
        </div>
        """
        )

    def _create_processing_interface(self, gr_state: gr.State) -> Dict[str, Any]:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Document Upload & Configuration")

                file_upload = gr.File(
                    label="Select Document",
                    file_types=self.config["supported_formats"],
                    type="binary",
                )

                with gr.Accordion("🔧 Processing Configuration", open=True):
                    llm_enabled_by_default = False

                    enable_llm = gr.Checkbox(
                        label="Enable Gemini AI Analysis",
                        value=llm_enabled_by_default,
                        info="Toggle to include Gemini-powered analysis in the processing pipeline",
                    )

                    llm_status = gr.Markdown(
                        value=self._build_llm_status_message(
                            llm_enabled_by_default,
                            bool(self.default_gemini_key),
                        )
                    )

                    with gr.Group(visible=llm_enabled_by_default) as llm_controls:
                        analysis_type = gr.Dropdown(
                            choices=[
                                ("Quality Analysis - Комплексна оцінка якості конвертації", "quality_analysis"),
                                ("Structure Review - Фокус на збереження ієрархії документа", "structure_review"),
                                ("Content Summary - Тематичний аналіз та ключові інсайти", "content_summary"),
                                ("Extraction Quality - Оцінка збереження даних", "extraction_quality"),
                            ],
                            value="quality_analysis",
                            label="Analysis Type",
                            interactive=True,
                        )

                        model_preference = gr.Dropdown(
                            choices=[
                                ("Gemini 2.0 Pro (Advanced Reasoning)", GeminiModel.PRO.value),
                                ("Gemini 2.0 Flash (Fast Inference)", GeminiModel.FLASH.value),
                                ("Gemini 2.5 Flash (Enhanced Quality)", GeminiModel.FLASH_25.value),
                                ("Gemini 1.5 Pro (Legacy)", GeminiModel.LEGACY_PRO.value),
                                ("Gemini 1.5 Flash (Legacy)", GeminiModel.LEGACY_FLASH.value),
                            ],
                            value=GeminiModel.PRO.value,
                            label="AI Model Preference",
                            interactive=True,
                        )

                        gemini_api_key = gr.Textbox(
                            label="Gemini API Key",
                            type="password",
                            value=self.default_gemini_key,
                            placeholder="Enter your Google Gemini API key...",
                            info="Key is read from .env by default. Provide a key here to override.",
                            interactive=True,
                        )

                with gr.Row():
                    process_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
                    clear_btn = gr.Button("🔄 Clear Session", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Processing Results")
                
                # 🚨 ДОДАНО ВАЖЛИВЕ ПОВІДОМЛЕННЯ ПРО ВИПРАВЛЕННЯ
                gr.HTML("""
                <div style="background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 10px 0; color: #0c5460;">🔧 Architectural Fix Applied</h4>
                    <p style="margin: 0; color: #0c5460;"><strong>Fixed Issue:</strong> Different analysis types now show different results in Markdown Output!</p>
                    <ul style="margin: 10px 0 0 20px; color: #0c5460;">
                        <li><strong>Quality Analysis:</strong> Shows detailed quality metrics and AI feedback</li>
                        <li><strong>Structure Review:</strong> Shows document structure analysis and organization</li>
                        <li><strong>Content Summary:</strong> Shows thematic analysis and key insights</li>
                        <li><strong>Extraction Quality:</strong> Shows data preservation assessment</li>
                    </ul>
                </div>
                """)
                
                status_display = gr.HTML()

                with gr.Tabs():
                    with gr.TabItem("📄 Original Document"):
                        original_preview = gr.HTML()

                    with gr.TabItem("📝 Markdown Output"):
                        gr.Markdown("**Результати обробки будуть показані тут з урахуванням обраного Analysis Type**")
                        markdown_output = gr.Code(
                            language="markdown",
                            show_label=False,
                            interactive=False,
                        )

                    with gr.TabItem("📈 Quick Metrics"):
                        quick_metrics = gr.JSON()

        return {
            "file_upload": file_upload,
            "enable_llm": enable_llm,
            "gemini_api_key": gemini_api_key,
            "analysis_type": analysis_type,
            "model_preference": model_preference,
            "llm_status": llm_status,
            "llm_controls": llm_controls,
            "process_btn": process_btn,
            "clear_btn": clear_btn,
            "status_display": status_display,
            "original_preview": original_preview,
            "markdown_output": markdown_output,
            "quick_metrics": quick_metrics,
        }

    def _build_llm_status_message(self, enabled: bool, key_present: bool) -> str:
        if not enabled:
            return "🔒 **AI Analysis Disabled.** Only conversion features are active."

        if key_present:
            return "🤖 **AI Analysis Enabled.** Gemini will execute using the configured API key."

        return (
            "⚠️ **AI Analysis Enabled but no API key detected.** "
            "Add `GEMINI_API_KEY` to `.env` or provide it in the interface."
        )

    def _handle_llm_toggle(self, use_llm: bool, current_key: str):
        current_key = current_key or ""

        if use_llm:
            resolved_key = current_key or self.default_gemini_key
            key_update = gr.update(value=resolved_key, visible=True, interactive=True)
            analysis_update = gr.update(visible=True, interactive=True)
            model_update = gr.update(visible=True, interactive=True)
            controls_update = gr.update(visible=True)
            status_message = self._build_llm_status_message(True, bool(resolved_key))
        else:
            key_update = gr.update(visible=False, interactive=False)
            analysis_update = gr.update(visible=False, interactive=False)
            model_update = gr.update(visible=False, interactive=False)
            controls_update = gr.update(visible=False)
            status_message = self._build_llm_status_message(False, False)

        return controls_update, key_update, analysis_update, model_update, status_message

    def _create_analytics_interface(self, gr_state: gr.State) -> Dict[str, Any]:
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
            "refresh_btn": refresh_btn,
            "quality_dashboard": quality_dashboard,
            "analysis_summary": analysis_summary,
            "structure_metrics": structure_metrics,
        }

    def _create_status_interface(self) -> None:
        gr.Markdown("### ⚙️ System Status & Health")

        with gr.Row():
            with gr.Column():
                system_health = gr.JSON(
                    label="System Health Metrics",
                    value=self._get_system_status(),
                )

            with gr.Column():
                processing_stats = gr.JSON(
                    label="Processing Statistics",
                    value=self.orchestrator.get_processing_status(),
                )

    def _create_application_footer(self) -> None:
        gr.HTML(
            """
        <div style=\"text-align: center; padding: 1rem; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 2rem;\">
            <p>Built with enterprise-grade architecture principles |
            <a href=\"https://github.com/microsoft/markitdown\">Microsoft MarkItDown</a> |
            <a href=\"https://ai.google.dev/\">Google Gemini</a></p>
            <p><strong>🔧 Critical Fix Applied:</strong> Different analysis types now show different results!</p>
        </div>
        """
        )

    def _wire_event_handlers(
        self,
        processing_components: Dict[str, Any],
        analytics_components: Dict[str, Any],
        gr_state: gr.State,
    ) -> None:
        processing_components["enable_llm"].change(
            fn=self._handle_llm_toggle,
            inputs=[
                processing_components["enable_llm"],
                processing_components["gemini_api_key"],
            ],
            outputs=[
                processing_components["llm_controls"],
                processing_components["gemini_api_key"],
                processing_components["analysis_type"],
                processing_components["model_preference"],
                processing_components["llm_status"],
            ],
        )

        processing_components["process_btn"].click(
            fn=self._handle_document_processing,
            inputs=[
                processing_components["file_upload"],
                processing_components["enable_llm"],
                processing_components["gemini_api_key"],
                processing_components["analysis_type"],
                processing_components["model_preference"],
                gr_state,
            ],
            outputs=[
                processing_components["status_display"],
                processing_components["original_preview"],
                processing_components["markdown_output"],
                processing_components["quick_metrics"],
                gr_state,
            ],
            show_progress="full",
        )

        processing_components["clear_btn"].click(
            fn=self._handle_session_clear,
            inputs=[gr_state],
            outputs=[
                processing_components["status_display"],
                processing_components["original_preview"],
                processing_components["markdown_output"],
                processing_components["quick_metrics"],
                gr_state,
            ],
        )

        analytics_components["refresh_btn"].click(
            fn=self._handle_analytics_refresh,
            inputs=[gr_state],
            outputs=[
                analytics_components["quality_dashboard"],
                analytics_components["analysis_summary"],
                analytics_components["structure_metrics"],
            ],
        )

    async def _handle_document_processing(
        self,
        file_obj,
        use_llm: bool,
        gemini_api_key: str,
        analysis_type: str,
        model_preference: str,
        current_state: ApplicationState,
    ) -> Tuple[str, str, str, JSONDict, ApplicationState]:
        if not file_obj:
            error_response = self.ui_factory.create_error_response(
                "No file uploaded. Please select a document to process."
            )
            return (*error_response, current_state)

        try:
            llm_enabled = bool(use_llm)
            resolved_api_key = None

            if llm_enabled:
                resolved_api_key = (gemini_api_key or self.default_gemini_key).strip()
                if not resolved_api_key:
                    warning_response = self.ui_factory.create_error_response(
                        "Gemini analysis is enabled, but no API key was found. Add `GEMINI_API_KEY` to your .env file or "
                        "enter a key in the interface, or disable AI analysis to continue with conversion only."
                    )
                    return (*warning_response, current_state)

            file_content = file_obj.read() if hasattr(file_obj, "read") else file_obj
            if isinstance(file_content, str):
                file_content = file_content.encode("utf-8")

            file_metadata = {
                "filename": getattr(file_obj, "name", "uploaded_file"),
                "size": len(file_content),
                "extension": Path(getattr(file_obj, "name", "file.txt")).suffix.lower(),
                "upload_timestamp": datetime.now().isoformat(),
            }

            processing_request = ProcessingRequest(
                file_content=file_content,
                file_metadata=file_metadata,
                gemini_api_key=resolved_api_key if llm_enabled else None,
                analysis_type=analysis_type,
                model_preference=model_preference,
                use_llm=llm_enabled,
                session_context={
                    "session_id": current_state.session_id,
                    "llm_enabled": llm_enabled,
                },
            )

            logger.info(
                "Queued processing request | LLM: %s | Analysis type input: %s | Model: %s",
                llm_enabled,
                analysis_type,
                model_preference,
            )

            processing_response = await self.orchestrator.process_document(processing_request)
            updated_state = current_state.add_processing_result(processing_response)

            if processing_response.success:
                ui_response = self.ui_factory.create_success_response(processing_response)
            else:
                ui_response = self.ui_factory.create_error_response(
                    processing_response.error_details,
                    processing_response.processing_metadata,
                )

            return (*ui_response, updated_state)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Event handler error: %s", exc)
            error_response = self.ui_factory.create_error_response(
                f"System error during processing: {exc}"
            )
            return (*error_response, current_state)

    def _handle_session_clear(
        self, current_state: ApplicationState
    ) -> Tuple[str, str, str, JSONDict, ApplicationState]:
        fresh_state = ApplicationState(session_id=datetime.now().isoformat())

        clear_html = """
        <div style=\"background: #e3f2fd; border: 1px solid #2196f3; color: #1976d2; \
                    padding: 15px; border-radius: 8px; margin: 10px 0;\">
            <h4 style=\"margin: 0;\">🔄 Session Cleared</h4>
            <p style=\"margin: 5px 0 0 0;\">Ready for new document processing.</p>
        </div>
        """

        return (clear_html, "", "", {}, fresh_state)

    def _handle_analytics_refresh(
        self, current_state: ApplicationState
    ) -> Tuple[Any, str, JSONDict]:
        import plotly.graph_objects as go

        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Process documents to see analytics",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        empty_fig.update_layout(title="Analytics Dashboard", height=400)

        if not current_state.processing_history:
            return (empty_fig, "*Process documents to see detailed analysis*", {})

        latest_result = next(
            (result for result in reversed(current_state.processing_history) if result.success),
            None,
        )

        if not latest_result:
            return (empty_fig, "*No successful processing results available*", {})

        try:
            quality_dashboard = self.ui_factory.viz_engine.create_quality_dashboard(
                latest_result.conversion_result,
                latest_result.analysis_result,
            )

            if latest_result.analysis_result:
                analysis_summary = self._format_analysis_summary(latest_result.analysis_result)
            else:
                analysis_summary = (
                    "**Basic conversion completed.** Add Gemini API key for AI-powered analysis."
                )

            structure_metrics = latest_result.quality_metrics.get("structural_metrics", {})

            return (quality_dashboard, analysis_summary, structure_metrics)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Analytics refresh error: %s", exc)
            return (empty_fig, f"*Analytics refresh failed: {exc}*", {"error": str(exc)})

    def _format_analysis_summary(self, analysis_result) -> str:
        if not analysis_result or not analysis_result.success:
            return "*AI analysis not available*"

        content = analysis_result.content
        analysis_type = analysis_result.analysis_type.value.replace("_", " ").title()

        summary = f"## 🤖 {analysis_type}\n\n"
        summary += f"**Model:** {analysis_result.model_used.value}  \n"
        summary += f"**Processing Time:** {analysis_result.processing_time:.2f}s\n\n"

        if "overall_score" in content:
            summary += "### 📊 Quality Assessment\n"
            summary += f"**Overall Score:** {content.get('overall_score', 0)}/10\n\n"

            scores = []
            if "structure_score" in content:
                scores.append(f"Structure: {content['structure_score']}/10")
            if "completeness_score" in content:
                scores.append(f"Completeness: {content['completeness_score']}/10")
            if "accuracy_score" in content:
                scores.append(f"Accuracy: {content['accuracy_score']}/10")

            if scores:
                summary += "**Detailed Scores:** " + " | ".join(scores) + "\n\n"

        if "executive_summary" in content:
            summary += f"### 📋 Executive Summary\n{content['executive_summary']}\n\n"

        if "detailed_feedback" in content:
            feedback = content["detailed_feedback"][:300]
            summary += (
                f"### 💡 Key Insights\n{feedback}"
                f"{'...' if len(content['detailed_feedback']) > 300 else ''}\n\n"
            )

        if content.get("recommendations"):
            summary += "### 🎯 Recommendations\n"
            for idx, rec in enumerate(content["recommendations"][:3], 1):
                summary += f"{idx}. {rec}\n"

        return summary

    def _get_system_status(self) -> JSONDict:
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            return {
                "system": {
                    "status": "Operational",
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "available_memory_gb": round(memory.available / (1024 ** 3), 2),
                    "platform": os.name,
                },
                "application": {
                    "version": self.config["version"],
                    "max_file_size_mb": self.config["max_file_size_mb"],
                },
            }

        except ImportError:
            return {
                "system": {
                    "status": "Degraded",
                    "cpu_usage_percent": None,
                    "memory_usage_percent": None,
                    "available_memory_gb": None,
                    "platform": os.name,
                },
                "application": {
                    "version": self.config["version"],
                    "max_file_size_mb": self.config["max_file_size_mb"],
                },
            }


class ApplicationFactory:
    """Factory helpers for constructing app instances."""

    @staticmethod
    def create_hf_spaces_app() -> MarkItDownTestingApp:
        logger.info("Initializing MarkItDown Testing Platform for HF Spaces deployment")

        processing_config = ProcessingConfig(
            max_file_size_mb=50,
            max_memory_usage_gb=12.0,
            processing_timeout=300,
            max_concurrent_processes=2,
        )

        resource_manager = ResourceManager(processing_config)
        file_handler = StreamlineFileHandler(resource_manager)
        conversion_engine = HFConversionEngine(resource_manager, processing_config)
        gemini_manager = GeminiConnectionManager()

        viz_config = VisualizationConfig(
            theme=VisualizationConfig.VisualizationTheme.CORPORATE,
            width=800,
            height=600,
        )
        viz_engine = InteractiveVisualizationEngine(viz_config)
        quality_calculator = QualityMetricsCalculator()

        orchestrator = DocumentProcessingOrchestrator(
            file_handler=file_handler,
            conversion_engine=conversion_engine,
            gemini_manager=gemini_manager,
            quality_calculator=quality_calculator,
        )

        ui_factory = GradioResponseFactory(viz_engine)

        app = MarkItDownTestingApp(
            orchestrator=orchestrator,
            ui_factory=ui_factory,
        )

        logger.info("Application initialized successfully - Ready for HF Spaces deployment")
        return app

    @staticmethod
    def create_local_development_app() -> MarkItDownTestingApp:
        processing_config = ProcessingConfig(
            max_file_size_mb=100,
            max_memory_usage_gb=32.0,
            processing_timeout=600,
            max_concurrent_processes=4,
        )

        logging.getLogger().setLevel(logging.DEBUG)

        resource_manager = ResourceManager(processing_config)
        file_handler = StreamlineFileHandler(resource_manager)
        conversion_engine = HFConversionEngine(resource_manager, processing_config)
        gemini_manager = GeminiConnectionManager()

        viz_config = VisualizationConfig(
            theme=VisualizationConfig.VisualizationTheme.CORPORATE,
            width=800,
            height=600,
        )
        viz_engine = InteractiveVisualizationEngine(viz_config)
        quality_calculator = QualityMetricsCalculator()

        orchestrator = DocumentProcessingOrchestrator(
            file_handler=file_handler,
            conversion_engine=conversion_engine,
            gemini_manager=gemini_manager,
            quality_calculator=quality_calculator,
        )

        ui_factory = GradioResponseFactory(viz_engine)

        return MarkItDownTestingApp(
            orchestrator=orchestrator,
            ui_factory=ui_factory,
        )


def setup_production_environment() -> None:
    os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp")
    os.environ.setdefault("HF_HOME", "/tmp")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import psutil

        memory = psutil.virtual_memory()
        logger.info(
            "Production environment initialized - Available memory: %.2f GB",
            memory.available / (1024 ** 3),
        )

        if memory.available < 2 * (1024 ** 3):
            logger.warning("Low memory detected - enabling aggressive cleanup policies")

    except ImportError:
        logger.warning("psutil not available - resource monitoring disabled")


def create_gradio_app() -> gr.Blocks:
    setup_production_environment()
    app = ApplicationFactory.create_hf_spaces_app()
    return app.create_interface()


def main() -> None:
    setup_production_environment()
    app = ApplicationFactory.create_hf_spaces_app()
    interface = app.create_interface()

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("PORT", 7860)),
        "share": False,
        "show_error": True,
        "max_file_size": f"{50 * 1024 * 1024}b",
        "allowed_paths": ["/tmp"],
        "root_path": os.environ.get("GRADIO_ROOT_PATH", ""),
    }

    logger.info("Launching MarkItDown Testing Platform on port %s", launch_kwargs["server_port"])
    interface.launch(**launch_kwargs)


__all__ = [
    "ApplicationFactory",
    "MarkItDownTestingApp", 
    "GradioResponseFactory",
    "ApplicationState",
    "create_gradio_app",
    "main",
]
