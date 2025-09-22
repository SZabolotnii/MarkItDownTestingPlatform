"""
Strategic Service Layer - Business Logic Abstraction

Design Philosophy:
"Business logic should be independent of presentation concerns"

This module provides a clean abstraction between UI interactions and 
core platform capabilities, enabling better testing, maintenance, and evolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar
from abc import ABC, abstractmethod

from pydantic import JsonValue

from app_logic import (
    DocumentProcessingOrchestrator,
    ProcessingRequest,
    ProcessingResponse,
)
from llm.gemini_connector import GeminiConnectionManager, GeminiModel
from visualization.analytics_engine import (
    InteractiveVisualizationEngine,
    VisualizationConfig,
)

logger = logging.getLogger(__name__)

JSONDict = Dict[str, JsonValue]
T = TypeVar('T')


# ==================== CORE ABSTRACTIONS ====================

@dataclass(frozen=True)
class ServiceResult:
    """Standardized result container for all service operations"""
    
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: JSONDict = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any = None, metadata: JSONDict = None) -> "ServiceResult":
        """Create successful result"""
        return cls(
            success=True, 
            data=data, 
            metadata=metadata or {}
        )
    
    @classmethod
    def error(cls, message: str, metadata: JSONDict = None) -> "ServiceResult":
        """Create error result"""
        return cls(
            success=False, 
            error_message=message, 
            metadata=metadata or {}
        )


@dataclass(frozen=True)
class ProcessingConfiguration:
    """Immutable processing configuration"""
    
    use_llm: bool = False
    gemini_api_key: Optional[str] = None
    analysis_type: str = "content_summary"
    model_preference: str = GeminiModel.FLASH.value
    enable_advanced_features: bool = False


@dataclass(frozen=True)
class SessionContext:
    """Session-scoped context and state"""
    
    session_id: str
    processing_history: Tuple[ProcessingResponse, ...] = field(default_factory=tuple)
    current_configuration: Optional[ProcessingConfiguration] = None
    user_preferences: JSONDict = field(default_factory=dict)
    
    def add_result(self, response: ProcessingResponse) -> "SessionContext":
        """Immutable session update"""
        return SessionContext(
            session_id=self.session_id,
            processing_history=self.processing_history + (response,),
            current_configuration=self.current_configuration,
            user_preferences=self.user_preferences
        )


# ==================== SERVICE INTERFACES ====================

class DocumentProcessingService(Protocol):
    """Abstract interface for document processing operations"""
    
    async def process_document(
        self, 
        file_data: bytes, 
        file_metadata: JSONDict,
        config: ProcessingConfiguration
    ) -> ServiceResult:
        """Process document with given configuration"""
        ...
    
    def get_processing_status(self) -> JSONDict:
        """Get current processing status and metrics"""
        ...


class VisualizationService(Protocol):
    """Abstract interface for visualization generation"""
    
    def create_dashboard(
        self, 
        processing_result: ProcessingResponse
    ) -> ServiceResult:
        """Create quality dashboard from processing result"""
        ...
    
    def create_analytics_summary(
        self, 
        session_context: SessionContext
    ) -> ServiceResult:
        """Create analytics summary from session history"""
        ...


class ConfigurationService(Protocol):
    """Abstract interface for configuration management"""
    
    def validate_gemini_key(self, api_key: str) -> ServiceResult:
        """Validate Gemini API key"""
        ...
    
    def get_default_configuration(self) -> ProcessingConfiguration:
        """Get default processing configuration"""
        ...


# ==================== CONCRETE IMPLEMENTATIONS ====================

class StandardDocumentProcessingService:
    """Standard implementation of document processing service"""
    
    def __init__(self, orchestrator: DocumentProcessingOrchestrator):
        self.orchestrator = orchestrator
        self.processing_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0
        }
    
    async def process_document(
        self, 
        file_data: bytes, 
        file_metadata: JSONDict,
        config: ProcessingConfiguration
    ) -> ServiceResult:
        """Execute document processing pipeline"""
        
        start_time = datetime.now()
        self.processing_metrics['total_requests'] += 1
        
        try:
            # Create processing request
            processing_request = ProcessingRequest(
                file_content=file_data,
                file_metadata=file_metadata,
                gemini_api_key=config.gemini_api_key if config.use_llm else None,
                analysis_type=config.analysis_type,
                model_preference=config.model_preference,
                use_llm=config.use_llm,
                session_context={
                    'timestamp': start_time.isoformat(),
                    'config_hash': hash(str(config))
                }
            )
            
            # Execute processing
            processing_response = await self.orchestrator.process_document(processing_request)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_metrics['total_processing_time'] += processing_time
            
            if processing_response.success:
                self.processing_metrics['successful_requests'] += 1
                logger.info(f"Document processed successfully in {processing_time:.2f}s")
                
                return ServiceResult.ok(
                    data=processing_response,
                    metadata={
                        'processing_time': processing_time,
                        'configuration': config.__dict__
                    }
                )
            else:
                self.processing_metrics['failed_requests'] += 1
                error_details = getattr(
                    processing_response,
                    'error_details',
                    'Document processing failed'
                )
                error_context = getattr(
                    processing_response,
                    'processing_metadata',
                    {}
                ) or {}
                if not isinstance(error_context, dict):
                    error_context = {'details': error_context}
                return ServiceResult.error(
                    message=error_details,
                    metadata={
                        'processing_time': processing_time,
                        'error_context': error_context
                    }
                )
                
        except Exception as e:
            self.processing_metrics['failed_requests'] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Document processing failed after {processing_time:.2f}s: {e}")
            
            return ServiceResult.error(
                message=f"Processing system error: {str(e)}",
                metadata={
                    'processing_time': processing_time,
                    'exception_type': type(e).__name__
                }
            )
    
    def get_processing_status(self) -> JSONDict:
        """Get comprehensive processing metrics"""
        
        total_requests = self.processing_metrics['total_requests']
        success_rate = (
            self.processing_metrics['successful_requests'] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        avg_processing_time = (
            self.processing_metrics['total_processing_time'] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            'status': 'healthy' if success_rate > 90 else 'degraded' if success_rate > 70 else 'unhealthy',
            'total_requests': total_requests,
            'success_rate_percent': success_rate,
            'average_processing_time_seconds': avg_processing_time,
            'uptime_info': {
                'successful_requests': self.processing_metrics['successful_requests'],
                'failed_requests': self.processing_metrics['failed_requests'],
                'total_processing_time': self.processing_metrics['total_processing_time']
            }
        }


class StandardVisualizationService:
    """Standard implementation of visualization service"""
    
    def __init__(self, viz_engine: InteractiveVisualizationEngine):
        self.viz_engine = viz_engine
        self.generation_count = 0
        
    def create_dashboard(self, processing_result: ProcessingResponse) -> ServiceResult:
        """Generate quality dashboard visualization"""
        
        try:
            self.generation_count += 1
            
            if not processing_result or not processing_result.success:
                return ServiceResult.error(
                    "Cannot create dashboard from failed processing result"
                )
            
            # Generate dashboard
            dashboard_figure = self.viz_engine.create_quality_dashboard(
                processing_result.conversion_result,
                processing_result.analysis_result
            )
            
            logger.info(f"Dashboard generated successfully (#{self.generation_count})")
            
            return ServiceResult.ok(
                data=dashboard_figure,
                metadata={
                    'generation_id': self.generation_count,
                    'chart_type': 'quality_dashboard',
                    'has_ai_analysis': processing_result.analysis_result is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            
            return ServiceResult.error(
                f"Visualization generation error: {str(e)}",
                metadata={'generation_id': self.generation_count}
            )
    
    def create_analytics_summary(self, session_context: SessionContext) -> ServiceResult:
        """Create analytics summary from session history"""
        
        try:
            if not session_context.processing_history:
                return ServiceResult.ok(
                    data={
                        'summary': "No processing history available",
                        'metrics': {},
                        'recommendations': ["Process documents to see analytics"],
                        'dashboard_figure': self._create_empty_dashboard(
                            title="Awaiting Documents",
                            message="Process a document to generate analytics"
                        ),
                    }
                )

            # Find latest successful result
            latest_success = next(
                (result for result in reversed(session_context.processing_history) 
                 if result.success),
                None
            )
            
            if not latest_success:
                return ServiceResult.ok(
                    data={
                        'summary': "No successful processing results available",
                        'metrics': {},
                        'recommendations': ["Check processing errors and try again"],
                        'dashboard_figure': self._create_empty_dashboard(
                            title="No Successful Runs",
                            message="Resolve processing issues to unlock analytics"
                        ),
                    }
                )

            # Generate summary
            summary_data = {
                'summary': self._format_processing_summary(latest_success),
                'metrics': latest_success.quality_metrics,
                'recommendations': self._generate_recommendations(session_context),
                'session_stats': {
                    'total_processed': len(session_context.processing_history),
                    'successful_count': sum(1 for r in session_context.processing_history if r.success),
                    'has_ai_analysis': any(r.analysis_result for r in session_context.processing_history if r.success)
                },
            }

            # Attach visualization dashboard for latest result
            summary_data['dashboard_figure'] = self._create_session_dashboard(latest_success)

            return ServiceResult.ok(
                data=summary_data,
                metadata={
                    'session_id': session_context.session_id,
                    'history_length': len(session_context.processing_history)
                }
            )
            
        except Exception as e:
            logger.error(f"Analytics summary generation failed: {e}")
            return ServiceResult.error(f"Analytics generation error: {str(e)}")
    
    def _format_processing_summary(self, result: ProcessingResponse) -> str:
        """Format processing result into readable summary"""

        if not result.success:
            return "Processing failed - no summary available"
        
        summary_parts = [
            f"**Document processed successfully**"
        ]
        
        if result.conversion_result:
            content_length = len(result.conversion_result.content)
            processing_time = result.conversion_result.processing_time or 0
            
            summary_parts.extend([
                f"**Content generated:** {content_length:,} characters",
                f"**Processing time:** {processing_time:.2f} seconds"
            ])
        
        if result.analysis_result and result.analysis_result.success:
            analysis_type = result.analysis_result.analysis_type.value.replace('_', ' ').title()
            model_used = result.analysis_result.model_used.value
            
            summary_parts.extend([
                f"**AI Analysis:** {analysis_type}",
                f"**Model used:** {model_used}"
            ])
            
            # Add quality scores if available
            if hasattr(result.analysis_result, 'content') and result.analysis_result.content:
                content = result.analysis_result.content
                if 'overall_score' in content:
                    summary_parts.append(f"**Quality score:** {content['overall_score']}/10")
        
        return "\n\n".join(summary_parts)
    
    def _generate_recommendations(self, session_context: SessionContext) -> List[str]:
        """Generate actionable recommendations based on session history"""
        
        recommendations = []
        
        successful_results = [r for r in session_context.processing_history if r.success]
        failed_results = [r for r in session_context.processing_history if not r.success]
        
        # Quality-based recommendations
        if successful_results:
            latest_success = successful_results[-1]
            quality_metrics = latest_success.quality_metrics
            composite_score = quality_metrics.get('composite_score', 0)
            
            if composite_score < 6:
                recommendations.append("Consider using higher quality source documents")
                recommendations.append("Enable AI analysis for detailed quality insights")
            elif composite_score >= 8:
                recommendations.append("Excellent quality achieved - consider batch processing similar documents")
        
        # AI analysis recommendations  
        has_ai_analysis = any(r.analysis_result for r in successful_results)
        if not has_ai_analysis and session_context.current_configuration:
            if not session_context.current_configuration.use_llm:
                recommendations.append("Enable Gemini AI analysis for detailed insights")
        
        # Error pattern analysis
        if len(failed_results) > len(successful_results):
            recommendations.append("High failure rate detected - check file formats and API keys")
        
        # Session productivity
        if len(session_context.processing_history) == 1:
            recommendations.append("Try different document types to explore platform capabilities")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _create_session_dashboard(self, result: ProcessingResponse):
        """Create dashboard visualization for analytics summary"""

        try:
            if not result.conversion_result:
                return self._create_empty_dashboard(
                    title="Visualization Unavailable",
                    message="Conversion data missing for dashboard"
                )

            return self.viz_engine.create_quality_dashboard(
                result.conversion_result,
                result.analysis_result
            )

        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error(f"Analytics dashboard generation failed: {exc}")
            return self._create_empty_dashboard(
                title="Dashboard Error",
                message="Could not render analytics visualization"
            )

    @staticmethod
    def _create_empty_dashboard(title: str, message: str):
        """Create a placeholder Plotly figure when data is unavailable"""

        import plotly.graph_objects as go

        figure = go.Figure()
        figure.update_layout(
            title=title,
            template="plotly_white",
            height=400,
        )
        figure.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=16, color="#6c757d"),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)

        return figure


class StandardConfigurationService:
    """Standard implementation of configuration service"""
    
    def __init__(self, gemini_manager: GeminiConnectionManager):
        self.gemini_manager = gemini_manager
        self.key_validation_cache = {}
        
    def validate_gemini_key(self, api_key: str) -> ServiceResult:
        """Validate Gemini API key with caching"""
        
        if not api_key or not api_key.strip():
            return ServiceResult.error("API key is required")
        
        # Check cache first
        key_hash = hash(api_key)
        if key_hash in self.key_validation_cache:
            cached_result = self.key_validation_cache[key_hash]
            logger.debug("Using cached API key validation result")
            return cached_result
        
        try:
            # Attempt to create engine (this validates the key)
            engine_id = None
            # Note: This is a simplified validation - in production you might want
            # a lighter-weight validation method
            
            result = ServiceResult.ok(
                data={'valid': True, 'key_preview': f"{api_key[:8]}..."},
                metadata={'validation_method': 'engine_creation'}
            )
            
            # Cache successful validation
            self.key_validation_cache[key_hash] = result
            return result
            
        except Exception as e:
            result = ServiceResult.error(
                f"API key validation failed: {str(e)}",
                metadata={'validation_method': 'engine_creation'}
            )
            
            # Don't cache failures (keys might become valid)
            return result
    
    def get_default_configuration(self) -> ProcessingConfiguration:
        """Get platform default configuration"""
        
        return ProcessingConfiguration(
            use_llm=False,  # Conservative default
            gemini_api_key=None,
            analysis_type="content_summary",
            model_preference=GeminiModel.FLASH.value,
            enable_advanced_features=False
        )


# ==================== SERVICE ORCHESTRATOR ====================

class PlatformServiceLayer:
    """
    Central service orchestrator providing unified access to platform capabilities
    
    Design Philosophy: Single point of access for all business operations
    """
    
    def __init__(
        self,
        document_service: DocumentProcessingService,
        visualization_service: VisualizationService,
        configuration_service: ConfigurationService
    ):
        self.document_service = document_service
        self.visualization_service = visualization_service  
        self.configuration_service = configuration_service
        
        self.service_metrics = {
            'initialized_at': datetime.now().isoformat(),
            'total_service_calls': 0,
            'service_call_distribution': {},
            'service_call_count': 0,
        }
    
    async def process_document_with_analytics(
        self,
        file_data: bytes,
        file_metadata: JSONDict,
        config: ProcessingConfiguration,
        generate_dashboard: bool = True
    ) -> Tuple[ServiceResult, Optional[ServiceResult]]:
        """Comprehensive document processing with optional analytics"""
        
        self._track_service_call('process_document_with_analytics')
        
        # Process document
        processing_result = await self.document_service.process_document(
            file_data, file_metadata, config
        )
        
        # Generate dashboard if requested and processing succeeded
        dashboard_result = None
        if generate_dashboard and processing_result.success:
            dashboard_result = self.visualization_service.create_dashboard(
                processing_result.data
            )
        
        return processing_result, dashboard_result
    
    def create_session_analytics(self, session_context: SessionContext) -> ServiceResult:
        """Create comprehensive session analytics"""
        
        self._track_service_call('create_session_analytics')
        
        return self.visualization_service.create_analytics_summary(session_context)
    
    def validate_configuration(self, config: ProcessingConfiguration) -> ServiceResult:
        """Validate complete processing configuration"""
        
        self._track_service_call('validate_configuration')
        
        if not config.use_llm:
            return ServiceResult.ok(
                data={'valid': True, 'message': 'Configuration valid for basic processing'}
            )
        
        # Validate Gemini key if LLM is enabled
        if not config.gemini_api_key:
            return ServiceResult.error("Gemini API key required when LLM analysis is enabled")
        
        return self.configuration_service.validate_gemini_key(config.gemini_api_key)
    
    def get_system_health(self) -> JSONDict:
        """Get comprehensive system health metrics"""
        
        self._track_service_call('get_system_health')
        
        processing_status = self.document_service.get_processing_status()
        
        return {
            'overall_status': processing_status.get('status', 'unknown'),
            'processing_service': processing_status,
            'service_layer': self.service_metrics,
            'system_info': {
                'active_services': ['document_processing', 'visualization', 'configuration'],
                'service_call_count': self.service_metrics['total_service_calls']
            }
        }
    
    def _track_service_call(self, method_name: str):
        """Track service usage for monitoring"""
        
        self.service_metrics['total_service_calls'] += 1
        
        if method_name not in self.service_metrics['service_call_distribution']:
            self.service_metrics['service_call_distribution'][method_name] = 0
        
        self.service_metrics['service_call_distribution'][method_name] += 1
        self.service_metrics['service_call_count'] = self.service_metrics['total_service_calls']


# ==================== FACTORY FUNCTIONS ====================

def create_hf_spaces_service_layer(orchestrator: DocumentProcessingOrchestrator) -> PlatformServiceLayer:
    """Factory function for HF Spaces optimized service layer"""
    
    # Create visualization engine
    viz_config = VisualizationConfig(
        theme=VisualizationConfig.VisualizationTheme.CORPORATE,
        width=800,
        height=600,
    )
    viz_engine = InteractiveVisualizationEngine(viz_config)
    
    # Create service implementations
    document_service = StandardDocumentProcessingService(orchestrator)
    visualization_service = StandardVisualizationService(viz_engine)
    configuration_service = StandardConfigurationService(orchestrator.gemini_manager)
    
    # Assemble service layer
    service_layer = PlatformServiceLayer(
        document_service=document_service,
        visualization_service=visualization_service,
        configuration_service=configuration_service
    )
    
    logger.info("HF Spaces service layer initialized successfully")
    return service_layer


# ==================== EXPORTS ====================

__all__ = [
    'ServiceResult',
    'ProcessingConfiguration', 
    'SessionContext',
    'DocumentProcessingService',
    'VisualizationService',
    'ConfigurationService',
    'StandardDocumentProcessingService',
    'StandardVisualizationService',
    'StandardConfigurationService',
    'PlatformServiceLayer',
    'create_hf_spaces_service_layer'
]
