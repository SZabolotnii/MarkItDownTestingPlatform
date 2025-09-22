"""
Strategic Event Orchestration Architecture - Centralized UI Event Management

Design Philosophy:
"Events should flow through a predictable, testable pipeline"

This module provides a systematic approach to UI event handling, separating 
concerns between user interactions, business logic execution, and UI updates.

Core Architectural Benefits:
- Predictable event flow patterns
- Centralized error handling and logging
- Simplified testing through clear interfaces
- Reduced coupling between UI components and business logic
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

from service_layer import (
    PlatformServiceLayer,
    ProcessingConfiguration,
    ServiceResult,
    SessionContext,
)
from response_factory import StrategicResponseFactory
from pydantic import JsonValue

logger = logging.getLogger(__name__)

JSONDict = Dict[str, JsonValue]
T = TypeVar('T')


# ==================== EVENT ABSTRACTIONS ====================

@dataclass
class EventContext:
    """Immutable context for event processing"""
    
    event_type: str
    timestamp: datetime
    session_context: SessionContext
    user_inputs: Dict[str, Any]
    metadata: JSONDict = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass
class EventResult:
    """Standardized result for event processing"""
    
    success: bool
    outputs: Tuple[Any, ...]
    updated_session: Optional[SessionContext]
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    @classmethod
    def ok(
        cls, 
        outputs: Tuple[Any, ...], 
        session: Optional[SessionContext] = None,
        processing_time: float = None
    ) -> "EventResult":
        """Create successful event result"""
        
        return cls(
            success=True,
            outputs=outputs,
            updated_session=session,
            processing_time=processing_time
        )
    
    @classmethod
    def error(
        cls, 
        message: str, 
        fallback_outputs: Tuple[Any, ...] = None,
        processing_time: float = None
    ) -> "EventResult":
        """Create error event result"""
        
        return cls(
            success=False,
            outputs=fallback_outputs or tuple(),
            updated_session=None,
            error_message=message,
            processing_time=processing_time
        )


# ==================== EVENT HANDLER INTERFACES ====================

class EventHandler(Protocol):
    """Abstract interface for all event handlers"""
    
    async def handle(self, context: EventContext) -> EventResult:
        """Handle event and return result"""
        ...


class EventValidator(Protocol):
    """Abstract interface for event validation"""
    
    def validate(self, context: EventContext) -> ServiceResult:
        """Validate event context before processing"""
        ...


class EventMiddleware(Protocol):
    """Abstract interface for event middleware"""
    
    async def process(
        self, 
        context: EventContext, 
        next_handler: Callable[[EventContext], Awaitable[EventResult]]
    ) -> EventResult:
        """Process event with middleware logic"""
        ...


# ==================== CONCRETE EVENT HANDLERS ====================

class DocumentProcessingEventHandler:
    """Specialized handler for document processing events"""
    
    def __init__(
        self, 
        service_layer: PlatformServiceLayer,
        response_factory: StrategicResponseFactory
    ):
        self.service_layer = service_layer
        self.response_factory = response_factory
        self.processing_count = 0
        
    async def handle(self, context: EventContext) -> EventResult:
        """Handle document processing event"""
        
        start_time = datetime.now()
        self.processing_count += 1
        
        try:
            # Extract inputs from context
            file_obj = context.user_inputs.get('file_obj')
            use_llm = context.user_inputs.get('use_llm', False)
            gemini_api_key = context.user_inputs.get('gemini_api_key', '')
            analysis_type = context.user_inputs.get('analysis_type', 'content_summary')
            model_preference = context.user_inputs.get('model_preference', 'gemini-2.0-flash-exp')
            
            # Validate inputs
            validation_result = self._validate_processing_inputs(
                file_obj, use_llm, gemini_api_key
            )
            if not validation_result.success:
                error_response = self.response_factory.create_processing_response(validation_result)
                return EventResult.error(
                    validation_result.error_message,
                    fallback_outputs=error_response,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Prepare processing configuration
            config = ProcessingConfiguration(
                use_llm=use_llm,
                gemini_api_key=gemini_api_key.strip() if gemini_api_key else None,
                analysis_type=analysis_type,
                model_preference=model_preference
            )
            
            # Extract file data and metadata
            file_content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            
            file_metadata = {
                'filename': getattr(file_obj, 'name', 'uploaded_file'),
                'size': len(file_content),
                'extension': self._extract_extension(getattr(file_obj, 'name', 'file.txt')),
                'upload_timestamp': datetime.now().isoformat(),
                'processing_id': self.processing_count
            }
            
            logger.info(
                f"Processing document #{self.processing_count} | "
                f"LLM: {use_llm} | Type: {analysis_type} | Model: {model_preference}"
            )
            
            # Execute processing through service layer
            processing_result, dashboard_result = await self.service_layer.process_document_with_analytics(
                file_content, file_metadata, config, generate_dashboard=True
            )
            
            # Create UI response
            ui_response = self.response_factory.create_processing_response(processing_result)
            
            # Update session context with result
            updated_session = None
            if processing_result.success and processing_result.data:
                updated_session = context.session_context.add_result(processing_result.data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Document processing completed in {processing_time:.2f}s | "
                f"Success: {processing_result.success}"
            )
            
            return EventResult.ok(
                outputs=ui_response,
                session=updated_session,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Document processing event failed: {str(e)}"
            
            logger.error(f"{error_message} (after {processing_time:.2f}s)")
            logger.debug(f"Processing error traceback: {traceback.format_exc()}")
            
            # Create error response through factory
            error_service_result = ServiceResult.error(error_message)
            error_response = self.response_factory.create_processing_response(error_service_result)
            
            return EventResult.error(
                error_message,
                fallback_outputs=error_response,
                processing_time=processing_time
            )
    
    def _validate_processing_inputs(
        self, 
        file_obj: Any, 
        use_llm: bool, 
        gemini_api_key: str
    ) -> ServiceResult:
        """Validate document processing inputs"""
        
        if not file_obj:
            return ServiceResult.error("No file uploaded. Please select a document to process.")
        
        if use_llm:
            api_key = gemini_api_key.strip() if gemini_api_key else ''
            if not api_key:
                return ServiceResult.error(
                    "Gemini analysis is enabled, but no API key was found. "
                    "Add `GEMINI_API_KEY` to your .env file or enter a key in the interface, "
                    "or disable AI analysis to continue with conversion only."
                )
        
        return ServiceResult.ok()
    
    def _extract_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        
        from pathlib import Path
        return Path(filename).suffix.lower() if filename else '.txt'


class AnalyticsEventHandler:
    """Specialized handler for analytics events"""
    
    def __init__(
        self, 
        service_layer: PlatformServiceLayer,
        response_factory: StrategicResponseFactory
    ):
        self.service_layer = service_layer
        self.response_factory = response_factory
        
    async def handle(self, context: EventContext) -> EventResult:
        """Handle analytics refresh event"""
        
        start_time = datetime.now()
        
        try:
            # Generate session analytics
            analytics_result = self.service_layer.create_session_analytics(context.session_context)
            
            # Create UI response
            ui_response = self.response_factory.create_analytics_response(analytics_result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.debug(f"Analytics refresh completed in {processing_time:.2f}s")
            
            return EventResult.ok(
                outputs=ui_response,
                session=context.session_context,  # Analytics don't modify session
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Analytics event failed: {str(e)}"
            
            logger.error(f"{error_message} (after {processing_time:.2f}s)")
            
            # Create empty analytics response
            error_result = ServiceResult.error(error_message)
            fallback_response = self.response_factory.create_analytics_response(error_result)
            
            return EventResult.error(
                error_message,
                fallback_outputs=fallback_response,
                processing_time=processing_time
            )


class ConfigurationEventHandler:
    """Specialized handler for configuration events"""
    
    def __init__(
        self, 
        service_layer: PlatformServiceLayer,
        response_factory: StrategicResponseFactory
    ):
        self.service_layer = service_layer
        self.response_factory = response_factory
        
    async def handle(self, context: EventContext) -> EventResult:
        """Handle configuration change events"""
        
        start_time = datetime.now()
        
        try:
            event_subtype = context.metadata.get('subtype', 'unknown')
            
            if event_subtype == 'llm_toggle':
                return await self._handle_llm_toggle(context)
            elif event_subtype == 'api_key_validation':
                return await self._handle_api_key_validation(context)
            else:
                return EventResult.error(f"Unknown configuration event subtype: {event_subtype}")
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Configuration event failed: {str(e)}"
            
            logger.error(f"{error_message} (after {processing_time:.2f}s)")
            
            return EventResult.error(
                error_message,
                processing_time=processing_time
            )
    
    async def _handle_llm_toggle(self, context: EventContext) -> EventResult:
        """Handle LLM enable/disable toggle"""
        
        use_llm = context.user_inputs.get('use_llm', False)
        current_key = context.user_inputs.get('current_key', '')
        default_key = context.metadata.get('default_key', '')
        
        if use_llm:
            resolved_key = current_key or default_key
            key_update = {'value': resolved_key, 'visible': True, 'interactive': True}
            analysis_update = {'visible': True, 'interactive': True}
            model_update = {'visible': True, 'interactive': True}
            controls_update = {'visible': True}
            status_message = "🤖 **AI Analysis Enabled.** Gemini will execute using the configured API key."
        else:
            key_update = {'visible': False, 'interactive': False}
            analysis_update = {'visible': False, 'interactive': False}
            model_update = {'visible': False, 'interactive': False}
            controls_update = {'visible': False}
            status_message = "🔒 **AI Analysis Disabled.** Only conversion features are active."
        
        # Return Gradio update objects
        import gradio as gr
        outputs = (
            gr.update(**controls_update),
            gr.update(**key_update),
            gr.update(**analysis_update),
            gr.update(**model_update),
            status_message
        )
        
        return EventResult.ok(outputs=outputs, session=context.session_context)
    
    async def _handle_api_key_validation(self, context: EventContext) -> EventResult:
        """Handle API key validation"""
        
        api_key = context.user_inputs.get('api_key', '')
        
        # Create processing configuration for validation
        config = ProcessingConfiguration(
            use_llm=bool(api_key),
            gemini_api_key=api_key
        )
        
        # Validate through service layer
        validation_result = self.service_layer.validate_configuration(config)
        
        # Create status message
        status_message = self.response_factory.create_validation_response(validation_result)
        
        return EventResult.ok(
            outputs=(status_message,),
            session=context.session_context
        )


class SessionEventHandler:
    """Specialized handler for session management events"""
    
    def __init__(self, response_factory: StrategicResponseFactory):
        self.response_factory = response_factory
        
    async def handle(self, context: EventContext) -> EventResult:
        """Handle session management events"""
        
        start_time = datetime.now()
        
        try:
            event_subtype = context.metadata.get('subtype', 'unknown')
            
            if event_subtype == 'clear':
                return await self._handle_session_clear(context)
            else:
                return EventResult.error(f"Unknown session event subtype: {event_subtype}")
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Session event failed: {str(e)}"
            
            logger.error(f"{error_message} (after {processing_time:.2f}s)")
            
            return EventResult.error(
                error_message,
                processing_time=processing_time
            )
    
    async def _handle_session_clear(self, context: EventContext) -> EventResult:
        """Handle session clear event"""
        
        # Create fresh session context
        fresh_session = SessionContext(session_id=datetime.now().isoformat())
        
        # Create clear response
        clear_response = self.response_factory.create_session_clear_response()
        
        logger.info(f"Session cleared - new session: {fresh_session.session_id}")
        
        return EventResult.ok(
            outputs=clear_response,
            session=fresh_session
        )


# ==================== EVENT MIDDLEWARE ====================

class LoggingMiddleware:
    """Middleware for event logging and monitoring"""
    
    def __init__(self):
        self.event_count = 0
        self.event_types = {}
        
    async def process(
        self, 
        context: EventContext, 
        next_handler: Callable[[EventContext], Awaitable[EventResult]]
    ) -> EventResult:
        """Process event with logging"""
        
        self.event_count += 1
        event_type = context.event_type
        
        # Track event types
        if event_type not in self.event_types:
            self.event_types[event_type] = 0
        self.event_types[event_type] += 1
        
        logger.debug(
            f"Processing event #{self.event_count} | "
            f"Type: {event_type} | Session: {context.session_context.session_id[:8]}..."
        )
        
        start_time = datetime.now()
        
        try:
            # Execute next handler
            result = await next_handler(context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            log_level = logging.INFO if result.success else logging.WARNING
            logger.log(
                log_level,
                f"Event completed | Type: {event_type} | "
                f"Success: {result.success} | Time: {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Event failed | Type: {event_type} | "
                f"Error: {str(e)} | Time: {processing_time:.2f}s"
            )
            raise


class ValidationMiddleware:
    """Middleware for event context validation"""
    
    async def process(
        self, 
        context: EventContext, 
        next_handler: Callable[[EventContext], Awaitable[EventResult]]
    ) -> EventResult:
        """Process event with validation"""
        
        # Validate context structure
        validation_result = self._validate_event_context(context)
        if not validation_result.success:
            return EventResult.error(f"Event validation failed: {validation_result.error_message}")
        
        # Execute next handler
        return await next_handler(context)
    
    def _validate_event_context(self, context: EventContext) -> ServiceResult:
        """Validate event context structure"""
        
        if not context.event_type:
            return ServiceResult.error("Event type is required")
        
        if not context.session_context:
            return ServiceResult.error("Session context is required")
        
        if not context.user_inputs:
            return ServiceResult.error("User inputs are required")
        
        return ServiceResult.ok()


# ==================== EVENT ORCHESTRATOR ====================

class EventOrchestrator:
    """
    Central event orchestration system
    
    Design Philosophy: Single point of control for all UI events
    """
    
    def __init__(
        self,
        service_layer: PlatformServiceLayer,
        response_factory: StrategicResponseFactory
    ):
        self.service_layer = service_layer
        self.response_factory = response_factory
        
        # Initialize event handlers
        self.handlers = {
            'document_processing': DocumentProcessingEventHandler(service_layer, response_factory),
            'analytics_refresh': AnalyticsEventHandler(service_layer, response_factory),
            'configuration_change': ConfigurationEventHandler(service_layer, response_factory),
            'session_management': SessionEventHandler(response_factory),
        }
        
        # Initialize middleware
        self.middleware = [
            ValidationMiddleware(),
            LoggingMiddleware(),
        ]
        
        # Orchestrator metrics
        self.orchestrator_metrics = {
            'initialized_at': datetime.now().isoformat(),
            'total_events_processed': 0,
            'events_by_type': {},
            'active_sessions': set()
        }
        
        logger.info("Event orchestrator initialized successfully")
    
    async def process_event(self, context: EventContext) -> EventResult:
        """Process event through middleware chain and handler"""
        
        self.orchestrator_metrics['total_events_processed'] += 1
        self.orchestrator_metrics['active_sessions'].add(context.session_context.session_id)
        
        # Track event types
        event_type = context.event_type
        if event_type not in self.orchestrator_metrics['events_by_type']:
            self.orchestrator_metrics['events_by_type'][event_type] = 0
        self.orchestrator_metrics['events_by_type'][event_type] += 1
        
        # Get appropriate handler
        handler = self.handlers.get(context.event_type)
        if not handler:
            return EventResult.error(f"No handler found for event type: {context.event_type}")
        
        # Build middleware chain
        async def execute_handler(ctx: EventContext) -> EventResult:
            return await handler.handle(ctx)
        
        # Apply middleware in reverse order
        handler_chain = execute_handler
        for middleware in reversed(self.middleware):
            current_handler = handler_chain
            handler_chain = lambda ctx, mw=middleware, nh=current_handler: mw.process(ctx, nh)
        
        # Execute through middleware chain
        try:
            return await handler_chain(context)
        except Exception as e:
            logger.error(f"Event orchestration failed: {str(e)}")
            return EventResult.error(f"Event processing system error: {str(e)}")
    
    def get_orchestrator_metrics(self) -> JSONDict:
        """Get comprehensive orchestrator metrics"""
        
        return {
            **self.orchestrator_metrics,
            'active_session_count': len(self.orchestrator_metrics['active_sessions']),
            'handlers_registered': list(self.handlers.keys()),
            'middleware_count': len(self.middleware)
        }


# ==================== CONVENIENCE FUNCTIONS ====================

def create_event_context(
    event_type: str,
    session_context: SessionContext,
    user_inputs: Dict[str, Any],
    metadata: JSONDict = None
) -> EventContext:
    """Convenience function for creating event contexts"""
    
    return EventContext(
        event_type=event_type,
        timestamp=datetime.now(),
        session_context=session_context,
        user_inputs=user_inputs,
        metadata=metadata or {}
    )


# ==================== EXPORTS ====================

__all__ = [
    'EventContext',
    'EventResult',
    'EventHandler',
    'EventValidator',
    'EventMiddleware',
    'DocumentProcessingEventHandler',
    'AnalyticsEventHandler',
    'ConfigurationEventHandler',
    'SessionEventHandler',
    'LoggingMiddleware',
    'ValidationMiddleware',
    'EventOrchestrator',
    'create_event_context',
]