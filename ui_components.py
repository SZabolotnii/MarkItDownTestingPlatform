"""
Strategic UI Component Architecture - Modular Interface Design

Design Philosophy:
"Each component should have a single, well-defined responsibility"

This module breaks down monolithic UI construction into focused, testable, and 
maintainable components that can be composed flexibly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import gradio as gr

from service_layer import (
    ProcessingConfiguration,
    ServiceResult,
    SessionContext,
)
from llm.gemini_connector import GeminiModel

logger = logging.getLogger(__name__)


# ==================== COMPONENT INTERFACES ====================

class UIComponent(Protocol):
    """Abstract interface for all UI components"""
    
    def create(self) -> Any:
        """Create the Gradio component"""
        ...
    
    def get_inputs(self) -> List[gr.components.Component]:
        """Get input components for event handling"""
        ...
    
    def get_outputs(self) -> List[gr.components.Component]:
        """Get output components for event handling"""
        ...


class EventHandler(Protocol):
    """Abstract interface for UI event handlers"""
    
    async def handle(self, *args) -> Tuple[Any, ...]:
        """Handle UI events and return outputs"""
        ...


# ==================== CONFIGURATION COMPONENTS ====================

@dataclass
class DocumentUploadConfig:
    """Configuration for document upload component"""
    
    max_file_size_mb: int = 50
    supported_formats: List[str] = None
    upload_label: str = "Select Document"
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                ".pdf", ".docx", ".pptx", ".xlsx", ".txt", 
                ".html", ".htm", ".csv", ".json", ".xml"
            ]


class DocumentUploadComponent:
    """Focused component for document upload functionality"""
    
    def __init__(self, config: DocumentUploadConfig):
        self.config = config
        self.file_upload = None
        
    def create(self) -> gr.File:
        """Create document upload interface"""
        
        self.file_upload = gr.File(
            label=self.config.upload_label,
            file_types=self.config.supported_formats,
            type="binary",
        )
        
        return self.file_upload
    
    def get_inputs(self) -> List[gr.components.Component]:
        return [self.file_upload] if self.file_upload else []
    
    def get_outputs(self) -> List[gr.components.Component]:
        return []  # Upload component doesn't have outputs


class AIConfigurationComponent:
    """Focused component for AI analysis configuration"""
    
    def __init__(self, default_api_key: str = ""):
        self.default_api_key = default_api_key
        self.enable_llm = None
        self.llm_status = None
        self.llm_controls = None
        self.analysis_type = None
        self.model_preference = None
        self.gemini_api_key = None
        
    def create(self) -> Dict[str, Any]:
        """Create AI configuration interface"""
        
        with gr.Accordion("🔧 Processing Configuration", open=True):
            llm_enabled_by_default = bool(self.default_api_key)
            
            self.enable_llm = gr.Checkbox(
                label="Enable Gemini AI Analysis",
                value=llm_enabled_by_default,
                info="Toggle to include Gemini-powered analysis in the processing pipeline",
            )
            
            self.llm_status = gr.Markdown(
                value=self._build_status_message(
                    llm_enabled_by_default,
                    bool(self.default_api_key),
                )
            )
            
            with gr.Group(visible=llm_enabled_by_default) as self.llm_controls:
                self.analysis_type = gr.Dropdown(
                    choices=[
                        ("Quality Analysis - Комплексна оцінка якості конвертації", "quality_analysis"),
                        ("Structure Review - Фокус на збереження ієрархії документа", "structure_review"),
                        ("Content Summary - Тематичний аналіз та ключові інсайти", "content_summary"),
                        ("Extraction Quality - Оцінка збереження даних", "extraction_quality"),
                    ],
                    value="content_summary",
                    label="Analysis Type",
                    interactive=True,
                )
                
                self.model_preference = gr.Dropdown(
                    choices=[
                        ("Gemini 2.0 Pro (Advanced Reasoning)", GeminiModel.PRO.value),
                        ("Gemini 2.0 Flash (Fast Inference)", GeminiModel.FLASH.value),
                        ("Gemini 2.5 Flash (Enhanced Quality)", GeminiModel.FLASH_25.value),
                        ("Gemini 1.5 Pro (Legacy)", GeminiModel.LEGACY_PRO.value),
                        ("Gemini 1.5 Flash (Legacy)", GeminiModel.LEGACY_FLASH.value),
                    ],
                    value=GeminiModel.FLASH.value,
                    label="AI Model Preference",
                    interactive=True,
                )
                
                self.gemini_api_key = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    value=self.default_api_key,
                    placeholder="Enter your Google Gemini API key...",
                    info="Key is read from .env by default. Provide a key here to override.",
                    interactive=True,
                )
        
        return {
            'enable_llm': self.enable_llm,
            'llm_status': self.llm_status,
            'llm_controls': self.llm_controls,
            'analysis_type': self.analysis_type,
            'model_preference': self.model_preference,
            'gemini_api_key': self.gemini_api_key,
        }
    
    def get_inputs(self) -> List[gr.components.Component]:
        components = []
        if self.enable_llm:
            components.append(self.enable_llm)
        if self.gemini_api_key:
            components.append(self.gemini_api_key)
        if self.analysis_type:
            components.append(self.analysis_type)
        if self.model_preference:
            components.append(self.model_preference)
        return components
    
    def get_outputs(self) -> List[gr.components.Component]:
        components = []
        if self.llm_status:
            components.append(self.llm_status)
        if self.llm_controls:
            components.append(self.llm_controls)
        return components
    
    def _build_status_message(self, enabled: bool, key_present: bool) -> str:
        """Build status message based on configuration state"""
        
        if not enabled:
            return "🔒 **AI Analysis Disabled.** Only conversion features are active."
        
        if key_present:
            return "🤖 **AI Analysis Enabled.** Gemini will execute using the configured API key."
        
        return (
            "⚠️ **AI Analysis Enabled but no API key detected.** "
            "Add `GEMINI_API_KEY` to `.env` or provide it in the interface."
        )
    
    def handle_toggle_change(self, use_llm: bool, current_key: str):
        """Handle LLM toggle state changes"""
        
        current_key = current_key or ""
        
        if use_llm:
            resolved_key = current_key or self.default_api_key
            key_update = gr.update(value=resolved_key, visible=True, interactive=True)
            analysis_update = gr.update(visible=True, interactive=True)
            model_update = gr.update(visible=True, interactive=True)
            controls_update = gr.update(visible=True)
            status_message = self._build_status_message(True, bool(resolved_key))
        else:
            key_update = gr.update(visible=False, interactive=False)
            analysis_update = gr.update(visible=False, interactive=False)
            model_update = gr.update(visible=False, interactive=False)
            controls_update = gr.update(visible=False)
            status_message = self._build_status_message(False, False)
        
        return controls_update, key_update, analysis_update, model_update, status_message


# ==================== PROCESSING INTERFACE COMPONENTS ====================

class ProcessingControlsComponent:
    """Focused component for processing action controls"""
    
    def __init__(self):
        self.process_btn = None
        self.clear_btn = None
        
    def create(self) -> Dict[str, gr.Button]:
        """Create processing control buttons"""
        
        with gr.Row():
            self.process_btn = gr.Button(
                "🚀 Process Document", 
                variant="primary", 
                size="lg"
            )
            self.clear_btn = gr.Button(
                "🔄 Clear Session", 
                variant="secondary"
            )
        
        return {
            'process_btn': self.process_btn,
            'clear_btn': self.clear_btn,
        }
    
    def get_inputs(self) -> List[gr.components.Component]:
        return [self.process_btn, self.clear_btn]
    
    def get_outputs(self) -> List[gr.components.Component]:
        return []  # Control components don't have outputs


class ResultsDisplayComponent:
    """Focused component for displaying processing results"""
    
    def __init__(self):
        self.status_display = None
        self.original_preview = None
        self.markdown_output = None
        self.quick_metrics = None
        
    def create(self) -> Dict[str, Any]:
        """Create results display interface"""
        
        gr.Markdown("### 📊 Processing Results")
        
        self.status_display = gr.HTML()
        
        with gr.Tabs():
            with gr.TabItem("📄 Original Document"):
                self.original_preview = gr.HTML()
            
            with gr.TabItem("📝 Markdown Output"):
                gr.Markdown("**Результати обробки будуть показані тут з урахуванням обраного Analysis Type**")
                self.markdown_output = gr.Code(
                    language="markdown",
                    show_label=False,
                    interactive=False,
                )
            
            with gr.TabItem("📈 Quick Metrics"):
                self.quick_metrics = gr.JSON()
        
        return {
            'status_display': self.status_display,
            'original_preview': self.original_preview,
            'markdown_output': self.markdown_output,
            'quick_metrics': self.quick_metrics,
        }
    
    def get_inputs(self) -> List[gr.components.Component]:
        return []  # Display components don't have inputs
    
    def get_outputs(self) -> List[gr.components.Component]:
        components = []
        if self.status_display:
            components.append(self.status_display)
        if self.original_preview:
            components.append(self.original_preview)
        if self.markdown_output:
            components.append(self.markdown_output)
        if self.quick_metrics:
            components.append(self.quick_metrics)
        return components


# ==================== ANALYTICS COMPONENTS ====================

class AnalyticsDashboardComponent:
    """Focused component for analytics dashboard"""
    
    def __init__(self):
        self.refresh_btn = None
        self.quality_dashboard = None
        self.analysis_summary = None
        self.structure_metrics = None
        
    def create(self) -> Dict[str, Any]:
        """Create analytics dashboard interface"""
        
        gr.Markdown("### 📊 Document Analysis Dashboard")
        
        with gr.Row():
            self.refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
        
        with gr.Row():
            self.quality_dashboard = gr.Plot(label="Quality Analysis Dashboard")
        
        with gr.Row():
            with gr.Column():
                self.analysis_summary = gr.Markdown("*Process a document to see analysis results*")
            with gr.Column():
                self.structure_metrics = gr.JSON(label="Structure Analysis")
        
        return {
            'refresh_btn': self.refresh_btn,
            'quality_dashboard': self.quality_dashboard,
            'analysis_summary': self.analysis_summary,
            'structure_metrics': self.structure_metrics,
        }
    
    def get_inputs(self) -> List[gr.components.Component]:
        return [self.refresh_btn] if self.refresh_btn else []
    
    def get_outputs(self) -> List[gr.components.Component]:
        components = []
        if self.quality_dashboard:
            components.append(self.quality_dashboard)
        if self.analysis_summary:
            components.append(self.analysis_summary)
        if self.structure_metrics:
            components.append(self.structure_metrics)
        return components


class SystemStatusComponent:
    """Focused component for system status monitoring"""
    
    def __init__(self):
        self.system_health = None
        self.processing_stats = None
        
    def create(self, health_data: Dict, stats_data: Dict) -> Dict[str, Any]:
        """Create system status interface"""
        
        gr.Markdown("### ⚙️ System Status & Health")
        
        with gr.Row():
            with gr.Column():
                self.system_health = gr.JSON(
                    label="System Health Metrics",
                    value=health_data,
                )
            
            with gr.Column():
                self.processing_stats = gr.JSON(
                    label="Processing Statistics",
                    value=stats_data,
                )
        
        return {
            'system_health': self.system_health,
            'processing_stats': self.processing_stats,
        }
    
    def get_inputs(self) -> List[gr.components.Component]:
        return []  # Status components typically don't have inputs
    
    def get_outputs(self) -> List[gr.components.Component]:
        components = []
        if self.system_health:
            components.append(self.system_health)
        if self.processing_stats:
            components.append(self.processing_stats)
        return components


# ==================== LAYOUT COMPONENTS ====================

class ApplicationHeaderComponent:
    """Focused component for application header"""
    
    def __init__(self, title: str, version: str):
        self.title = title
        self.version = version
        
    def create(self) -> gr.HTML:
        """Create application header"""
        
        return gr.HTML(
            f"""
            <div style=\"text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); \
                        color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;\">
                <h1 style=\"margin: 0; font-size: 2.5em;\">🚀 {self.title}</h1>
                <p style=\"margin: 10px 0; font-size: 1.2em;\">Enterprise-Grade Document Conversion Testing with AI-Powered Analysis</p>
                <p style=\"margin: 0; opacity: 0.9;\">
                    <em>Version {self.version} | Powered by Microsoft MarkItDown & Google Gemini</em>
                </p>
            </div>
            """
        )


class ApplicationFooterComponent:
    """Focused component for application footer"""
    
    def create(self) -> gr.HTML:
        """Create application footer"""
        
        return gr.HTML(
            """
            <div style=\"text-align: center; padding: 1rem; color: #6c757d; border-top: 1px solid #dee2e6; margin-top: 2rem;\">
                <p>Built with enterprise-grade architecture principles |
                <a href=\"https://github.com/microsoft/markitdown\">Microsoft MarkItDown</a> |
                <a href=\"https://ai.google.dev/\">Google Gemini</a></p>
                <p><strong>🔧 Strategic Refactoring Applied:</strong> Modular architecture with service layer separation!</p>
            </div>
            """
        )


# ==================== COMPOSITE COMPONENTS ====================

class ProcessingTabComponent:
    """Composite component that assembles the complete processing interface"""
    
    def __init__(self, upload_config: DocumentUploadConfig, default_api_key: str = ""):
        self.upload_config = upload_config
        self.default_api_key = default_api_key
        
        # Sub-components
        self.upload_component = DocumentUploadComponent(upload_config)
        self.ai_config_component = AIConfigurationComponent(default_api_key)
        self.processing_controls = ProcessingControlsComponent()
        self.results_display = ResultsDisplayComponent()
        
        # Component references for external access
        self.components = {}
        
    def create(self) -> Dict[str, Any]:
        """Create complete processing interface"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Document Upload & Configuration")
                
                # Document upload
                file_upload = self.upload_component.create()
                
                # AI configuration
                ai_components = self.ai_config_component.create()
                
                # Processing controls
                control_components = self.processing_controls.create()
            
            with gr.Column(scale=2):
                # Results display
                result_components = self.results_display.create()
        
        # Collect all components for external access
        self.components = {
            'file_upload': file_upload,
            **ai_components,
            **control_components,
            **result_components,
        }
        
        return self.components
    
    def get_all_inputs(self) -> List[gr.components.Component]:
        """Get all input components from sub-components"""
        
        inputs = []
        inputs.extend(self.upload_component.get_inputs())
        inputs.extend(self.ai_config_component.get_inputs())
        inputs.extend(self.processing_controls.get_inputs())
        return inputs
    
    def get_all_outputs(self) -> List[gr.components.Component]:
        """Get all output components from sub-components"""
        
        outputs = []
        outputs.extend(self.ai_config_component.get_outputs())
        outputs.extend(self.results_display.get_outputs())
        return outputs


class AnalyticsTabComponent:
    """Composite component that assembles the complete analytics interface"""
    
    def __init__(self):
        self.dashboard_component = AnalyticsDashboardComponent()
        self.components = {}
        
    def create(self) -> Dict[str, Any]:
        """Create complete analytics interface"""
        
        self.components = self.dashboard_component.create()
        return self.components
    
    def get_all_inputs(self) -> List[gr.components.Component]:
        """Get all input components"""
        return self.dashboard_component.get_inputs()
    
    def get_all_outputs(self) -> List[gr.components.Component]:
        """Get all output components"""
        return self.dashboard_component.get_outputs()


class SystemStatusTabComponent:
    """Composite component that assembles the system status interface"""
    
    def __init__(self, health_data: Dict, stats_data: Dict):
        self.health_data = health_data
        self.stats_data = stats_data
        self.status_component = SystemStatusComponent()
        self.components = {}
        
    def create(self) -> Dict[str, Any]:
        """Create complete system status interface"""
        
        self.components = self.status_component.create(
            self.health_data, 
            self.stats_data
        )
        return self.components


# ==================== COMPONENT FACTORY ====================

class UIComponentFactory:
    """Factory for creating UI components with consistent configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_processing_tab(self) -> ProcessingTabComponent:
        """Create processing tab with factory configuration"""
        
        upload_config = DocumentUploadConfig(
            max_file_size_mb=self.config.get('max_file_size_mb', 50),
            supported_formats=self.config.get('supported_formats'),
        )
        
        return ProcessingTabComponent(
            upload_config=upload_config,
            default_api_key=self.config.get('default_gemini_key', '')
        )
    
    def create_analytics_tab(self) -> AnalyticsTabComponent:
        """Create analytics tab"""
        return AnalyticsTabComponent()
    
    def create_system_status_tab(self, health_data: Dict, stats_data: Dict) -> SystemStatusTabComponent:
        """Create system status tab with provided data"""
        return SystemStatusTabComponent(health_data, stats_data)
    
    def create_header(self) -> ApplicationHeaderComponent:
        """Create application header"""
        return ApplicationHeaderComponent(
            title=self.config.get('title', 'MarkItDown Testing Platform'),
            version=self.config.get('version', '2.0.0-enterprise')
        )
    
    def create_footer(self) -> ApplicationFooterComponent:
        """Create application footer"""
        return ApplicationFooterComponent()


# ==================== EXPORTS ====================

__all__ = [
    'UIComponent',
    'EventHandler',
    'DocumentUploadConfig',
    'DocumentUploadComponent',
    'AIConfigurationComponent',
    'ProcessingControlsComponent',
    'ResultsDisplayComponent',
    'AnalyticsDashboardComponent',
    'SystemStatusComponent',
    'ApplicationHeaderComponent',
    'ApplicationFooterComponent',
    'ProcessingTabComponent',
    'AnalyticsTabComponent',
    'SystemStatusTabComponent',
    'UIComponentFactory',
]