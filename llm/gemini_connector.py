"""
Enterprise-Grade Gemini Integration Layer

Strategic Design Philosophy:
- Multi-model orchestration for diverse analysis needs
- Robust error handling with graceful degradation
- Configurable analysis pipelines for different use cases
- Performance optimization for HF Spaces constraints

This module provides a comprehensive Gemini API integration designed for
enterprise-scale document analysis with focus on reliability and extensibility.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum

from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, validator, JsonValue


JSONDict = Dict[str, JsonValue]


# Strategic Configuration Classes
class AnalysisType(Enum):
    """Enumeration of available analysis types"""
    QUALITY_ANALYSIS = "quality_analysis"
    STRUCTURE_REVIEW = "structure_review"
    CONTENT_SUMMARY = "content_summary"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EXTRACTION_QUALITY = "extraction_quality"


class GeminiModel(Enum):
    """Available Gemini models with strategic use case mapping"""

    PRO = "gemini-2.0-pro-exp"              # Latest high-accuracy reasoning model
    FLASH = "gemini-2.0-flash-exp"          # Latest high-speed model
    FLASH_25 = "gemini-2.5-flash"           # Enhanced quality flash model
    LEGACY_PRO = "gemini-1.5-pro"           # Legacy compatibility
    LEGACY_FLASH = "gemini-1.5-flash"       # Legacy compatibility
    PRO_VISION = "gemini-1.5-pro-vision"    # Multimodal content analysis

    @classmethod
    def from_str(cls, value: Union[str, "GeminiModel", None]) -> "GeminiModel":
        """Resolve string input to an enum member with graceful fallbacks"""

        if isinstance(value, cls):
            return value

        if value in (None, ""):
            return cls.PRO

        try:
            return cls(value)
        except ValueError as exc:
            legacy_aliases = {
                "gemini-1.5-pro": cls.LEGACY_PRO,
                "gemini-1.5-flash": cls.LEGACY_FLASH,
                "gemini-1.5-pro-vision": cls.PRO_VISION,
            }

            if value in legacy_aliases:
                return legacy_aliases[value]

            raise ValueError(f"Unsupported Gemini model: {value}") from exc


@dataclass
class GeminiConfig:
    """Comprehensive Gemini API configuration"""
    api_key: Optional[str] = None
    default_model: GeminiModel = GeminiModel.PRO
    max_tokens: int = 8192
    temperature: float = 0.1  # Low temperature for consistent analysis
    timeout_seconds: int = 60
    max_retry_attempts: int = 3
    safety_settings: Optional[List[types.SafetySetting]] = None
    
    def __post_init__(self):
        if self.safety_settings is None:
            self.safety_settings = [
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
            ]


class AnalysisRequest(BaseModel):
    """Structured request for document analysis"""
    
    content: str = Field(..., description="Markdown content to analyze")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    model: GeminiModel = Field(default=GeminiModel.PRO, description="Gemini model to use")
    custom_instructions: Optional[str] = Field(None, description="Additional analysis instructions")
    context: Optional[JSONDict] = Field(default_factory=dict, description="Additional context")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v

    @validator('model', pre=True, always=True)
    def validate_model(cls, value):
        return GeminiModel.from_str(value)


class AnalysisResponse(BaseModel):
    """Standardized analysis response structure"""
    
    success: bool
    analysis_type: AnalysisType
    model_used: GeminiModel
    content: JSONDict
    metadata: JSONDict
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None


class GeminiAnalysisEngine:
    """
    Comprehensive Gemini-powered analysis system
    
    Strategic Architecture:
    - Multi-model orchestration for optimal performance vs cost
    - Prompt engineering templates for consistent results
    - Error handling with intelligent retry mechanisms
    - Performance monitoring and optimization
    """
    
    # Strategic Prompt Templates for Different Analysis Types
    ANALYSIS_PROMPTS = {
        AnalysisType.QUALITY_ANALYSIS: {
            "system": """You are an expert document conversion analyst specializing in evaluating 
            the quality of document-to-Markdown conversions.""",
            "template": """
            Analyze the quality of this Markdown conversion from a document.
            
            **Analysis Focus Areas:**
            1. **Structure Preservation**: How well are headers, lists, tables maintained?
            2. **Content Completeness**: Is all information preserved from the original?
            3. **Formatting Accuracy**: Are formatting elements correctly converted?
            4. **Information Hierarchy**: Is the document structure logical and clear?
            5. **Readability**: How accessible is the converted content?
            
            **Content to Analyze:**
            ```markdown
            {content}
            ```
            
            **Provide your analysis as a structured JSON response with these fields:**
            - overall_score: (1-10 scale)
            - structure_score: (1-10 scale)
            - completeness_score: (1-10 scale)  
            - accuracy_score: (1-10 scale)
            - readability_score: (1-10 scale)
            - detailed_feedback: (string with specific observations)
            - recommendations: (array of improvement suggestions)
            - detected_elements: (object listing found structural elements)
            
            Focus on actionable insights and specific examples from the content.
            """,
        },
        
        AnalysisType.STRUCTURE_REVIEW: {
            "system": """You are a document structure specialist analyzing Markdown 
            document organization and hierarchy.""",
            "template": """
            Conduct a comprehensive structural analysis of this Markdown document.
            
            **Structure Analysis Requirements:**
            1. **Hierarchy Analysis**: Map all heading levels (H1, H2, H3, etc.)
            2. **List Structures**: Identify and categorize all lists (ordered, unordered, nested)
            3. **Table Analysis**: Evaluate table formatting and completeness
            4. **Content Organization**: Assess logical flow and organization
            5. **Special Elements**: Identify code blocks, links, images, etc.
            
            **Content to Analyze:**
            ```markdown
            {content}
            ```
            
            **Provide a structured JSON response with:**
            - document_outline: (hierarchical structure map)
            - heading_analysis: (object with heading counts and levels)
            - list_analysis: (detailed list structure information)
            - table_analysis: (table count, structure, formatting quality)
            - special_elements: (code blocks, links, images, etc.)
            - organization_score: (1-10 scale)
            - structure_recommendations: (array of specific improvements)
            - accessibility_notes: (readability and navigation considerations)
            
            Provide specific examples and actionable structural insights.
            """,
        },
        
        AnalysisType.CONTENT_SUMMARY: {
            "system": """You are a content analysis expert specializing in document 
            summarization and thematic analysis.""",
            "template": """
            Create a comprehensive content summary and thematic analysis of this document.
            
            **Summary Requirements:**
            1. **Executive Summary**: 2-3 sentence overview of main content
            2. **Key Topics**: Primary themes and subjects covered
            3. **Content Classification**: Document type, purpose, target audience
            4. **Information Density**: Assessment of content richness and depth
            5. **Actionable Insights**: Key takeaways and important information
            
            **Content to Analyze:**
            ```markdown
            {content}
            ```
            
            **Provide a structured JSON response with:**
            - executive_summary: (brief overview)
            - main_topics: (array of key themes)
            - document_classification: (type, purpose, audience)
            - content_metrics: (word count estimates, complexity level)
            - key_information: (array of important facts/insights)
            - content_quality: (1-10 scale for informativeness)
            - summary_recommendations: (suggestions for content improvement)
            - thematic_analysis: (deeper dive into content themes)
            
            Focus on extracting actionable intelligence from the content.
            """,
        },
        
        AnalysisType.EXTRACTION_QUALITY: {
            "system": """You are a data extraction quality specialist evaluating how well 
            information was preserved during document conversion.""",
            "template": """
            Evaluate the extraction quality and information preservation in this converted document.
            
            **Quality Assessment Areas:**
            1. **Data Preservation**: Are numbers, dates, names preserved accurately?
            2. **Formatting Retention**: How well were original formatting cues maintained?
            3. **Context Preservation**: Is the meaning and context clear?
            4. **Information Completeness**: Are there signs of missing information?
            5. **Conversion Artifacts**: Any obvious conversion errors or artifacts?
            
            **Content to Analyze:**
            ```markdown
            {content}
            ```
            
            **Provide a structured JSON response with:**
            - extraction_score: (1-10 overall quality)
            - data_accuracy: (assessment of numerical/factual data)
            - context_preservation: (meaning and relationships maintained)
            - formatting_quality: (original structure maintained)
            - completeness_indicators: (signs of missing content)
            - conversion_artifacts: (errors or issues detected)
            - quality_recommendations: (specific improvement suggestions)
            - confidence_level: (confidence in the analysis)
            
            Identify specific examples of good and poor extraction quality.
            """,
        }
    }
    
    def __init__(self, config: GeminiConfig):
        """Initialize Gemini Analysis Engine with configuration"""
        
        self.config = config
        self.client: Optional[genai.Client] = None
        self._initialize_client()
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
    
    def _initialize_client(self):
        """Initialize Gemini client with error handling"""
        
        if not self.config.api_key:
            raise ValueError("Gemini API key is required")
        
        try:
            self.client = genai.Client(api_key=self.config.api_key)

            # Optional warm-up to validate credentials without incurring generation cost
            try:
                _ = next(self.client.models.list(page_size=1), None)
            except Exception as list_error:  # pragma: no cover - defensive logging
                logging.debug(f"Model listing skipped: {list_error}")

            logging.info("Gemini client (google-genai) initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def analyze_content(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Execute comprehensive content analysis with retry logic
        
        Strategic Processing Approach:
        1. Validate request and prepare prompt
        2. Execute analysis with appropriate model
        3. Parse and validate response
        4. Return structured results with metadata
        """
        
        start_time = datetime.now()
        self.request_count += 1
        
        try:
            # Prepare analysis prompt
            prompt = self._build_analysis_prompt(request)

            # Select optimal model for analysis type
            model_enum = self._select_optimal_model(request.analysis_type, request.model)

            # Execute analysis
            response_text = await self._execute_analysis(model_enum.value, prompt)

            # Parse and structure response
            analysis_content = self._parse_analysis_response(response_text, request.analysis_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            return AnalysisResponse(
                success=True,
                analysis_type=request.analysis_type,
                model_used=model_enum,
                content=analysis_content,
                metadata={
                    'processing_time': processing_time,
                    'content_length': len(request.content),
                    'prompt_tokens': len(prompt.split()),  # Rough estimate
                    'timestamp': start_time.isoformat(),
                    'request_id': self.request_count
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logging.error(f"Analysis failed for {request.analysis_type}: {e}")
            
            return AnalysisResponse(
                success=False,
                analysis_type=request.analysis_type,
                model_used=request.model,
                content={},
                metadata={'error_timestamp': datetime.now().isoformat()},
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _build_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Build comprehensive analysis prompt from template"""
        
        prompt_config = self.ANALYSIS_PROMPTS.get(request.analysis_type)
        if not prompt_config:
            raise ValueError(f"Unsupported analysis type: {request.analysis_type}")
        
        # Build complete prompt with system context
        system_context = prompt_config["system"]
        main_prompt = prompt_config["template"].format(content=request.content)
        
        # Add custom instructions if provided
        if request.custom_instructions:
            main_prompt += f"\n\n**Additional Instructions:**\n{request.custom_instructions}"
        
        # Add context if available
        if request.context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in request.context.items()])
            main_prompt += f"\n\n**Context:**\n{context_str}"
        
        return f"{system_context}\n\n{main_prompt}"
    
    def _select_optimal_model(self, analysis_type: AnalysisType, requested_model: GeminiModel) -> GeminiModel:
        """Select optimal Gemini model based on analysis requirements"""

        # Strategic model selection based on analysis complexity
        model_recommendations = {
            AnalysisType.QUALITY_ANALYSIS: GeminiModel.PRO,      # Complex reasoning
            AnalysisType.STRUCTURE_REVIEW: GeminiModel.PRO,     # Detailed analysis
            AnalysisType.CONTENT_SUMMARY: GeminiModel.FLASH,    # Fast processing
            AnalysisType.COMPARATIVE_ANALYSIS: GeminiModel.PRO, # Complex comparison
            AnalysisType.EXTRACTION_QUALITY: GeminiModel.PRO,   # Detailed quality assessment
        }

        # Respect explicit model choices outside default presets
        default_overrides = {GeminiModel.PRO, GeminiModel.FLASH}
        if requested_model not in default_overrides:
            return requested_model

        return model_recommendations.get(analysis_type, requested_model)
    
    async def _execute_analysis(self, model_name: str, prompt: str) -> str:
        """Execute analysis using Gemini API with timeout and error handling"""

        if not self.client:
            raise RuntimeError("Gemini client is not initialized")

        def _run_generation() -> str:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            config_kwargs = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }

            if self.config.safety_settings:
                config_kwargs["safety_settings"] = self.config.safety_settings

            generation_config = types.GenerateContentConfig(**config_kwargs)

            try:
                stream = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

                collected_chunks: List[str] = []
                for chunk in stream:
                    text_part = getattr(chunk, "text", None)
                    if text_part:
                        collected_chunks.append(text_part)

                return "".join(collected_chunks)

            except AttributeError:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )
                return getattr(response, "text", getattr(response, "output_text", ""))

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_generation),
                timeout=self.config.timeout_seconds,
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"Gemini API request timed out after {self.config.timeout_seconds} seconds")
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def _parse_analysis_response(self, response_text: str, analysis_type: AnalysisType) -> JSONDict:
        """Parse and validate Gemini response into structured format"""
        
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                parsed_response = json.loads(json_content)
                
                # Validate required fields based on analysis type
                validated_response = self._validate_response_structure(parsed_response, analysis_type)
                return validated_response
            
            else:
                # Fallback: structure unstructured response
                return self._structure_unstructured_response(response_text, analysis_type)
                
        except json.JSONDecodeError:
            # Handle non-JSON response
            return self._structure_unstructured_response(response_text, analysis_type)
    
    def _validate_response_structure(self, response: JSONDict, analysis_type: AnalysisType) -> JSONDict:
        """Validate and ensure response contains required fields"""
        
        # Define required fields for each analysis type
        required_fields = {
            AnalysisType.QUALITY_ANALYSIS: [
                'overall_score', 'structure_score', 'completeness_score', 
                'accuracy_score', 'readability_score', 'detailed_feedback'
            ],
            AnalysisType.STRUCTURE_REVIEW: [
                'document_outline', 'heading_analysis', 'organization_score'
            ],
            AnalysisType.CONTENT_SUMMARY: [
                'executive_summary', 'main_topics', 'content_quality'
            ],
            AnalysisType.EXTRACTION_QUALITY: [
                'extraction_score', 'data_accuracy', 'completeness_indicators'
            ]
        }
        
        expected_fields = required_fields.get(analysis_type, [])
        
        # Ensure all required fields are present with defaults
        validated_response = response.copy()
        for field in expected_fields:
            if field not in validated_response:
                validated_response[field] = self._get_default_field_value(field)
        
        return validated_response
    
    def _get_default_field_value(self, field_name: str) -> Any:
        """Get default value for missing response fields"""
        
        if field_name.endswith('_score'):
            return 0
        elif field_name in ['detailed_feedback', 'executive_summary']:
            return "Analysis incomplete - field not provided"
        elif field_name.endswith('_analysis') or field_name == 'document_outline':
            return {}
        elif field_name in ['main_topics', 'recommendations']:
            return []
        else:
            return None
    
    def _structure_unstructured_response(self, response_text: str, analysis_type: AnalysisType) -> JSONDict:
        """Structure unstructured response text into expected format"""
        
        # Basic structuring based on analysis type
        base_structure = {
            'raw_response': response_text,
            'structured': False,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Add type-specific default structure
        if analysis_type == AnalysisType.QUALITY_ANALYSIS:
            base_structure.update({
                'overall_score': 5,  # Neutral default
                'detailed_feedback': response_text,
                'recommendations': []
            })
        elif analysis_type == AnalysisType.CONTENT_SUMMARY:
            base_structure.update({
                'executive_summary': response_text[:200] + "..." if len(response_text) > 200 else response_text,
                'content_quality': 5
            })
        
        return base_structure
    
    async def batch_analyze(self, requests: List[AnalysisRequest]) -> List[AnalysisResponse]:
        """Execute multiple analyses concurrently with rate limiting"""
        
        # Implement concurrent processing with semaphore for rate limiting
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        
        async def limited_analyze(request):
            async with semaphore:
                return await self.analyze_content(request)
        
        # Execute all requests concurrently
        tasks = [limited_analyze(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_response = AnalysisResponse(
                    success=False,
                    analysis_type=requests[i].analysis_type,
                    model_used=requests[i].model,
                    content={},
                    metadata={'batch_error': True},
                    error_message=str(result)
                )
                processed_results.append(error_response)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_metrics(self) -> JSONDict:
        """Get comprehensive performance metrics"""
        
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        success_rate = (
            (self.request_count - self.error_count) / self.request_count * 100
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'success_rate_percent': success_rate,
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'requests_per_minute': self.request_count / max(self.total_processing_time / 60, 1)
        }


class GeminiConnectionManager:
    """
    Enterprise-grade connection and configuration management for Gemini
    
    Strategic Features:
    - API key validation and secure storage
    - Connection health monitoring
    - Automatic reconnection and failover
    - Usage tracking and optimization recommendations
    """
    
    def __init__(self):
        self.engines: Dict[str, GeminiAnalysisEngine] = {}
        self.connection_health = {}
    
    async def create_engine(self, api_key: str, config: Optional[GeminiConfig] = None) -> str:
        """Create and validate new Gemini engine instance"""
        
        if not api_key or not api_key.strip():
            raise ValueError("Valid API key is required")
        
        # Create configuration
        if config is None:
            config = GeminiConfig(api_key=api_key)
        else:
            config.api_key = api_key
        
        # Generate unique engine ID
        engine_id = f"gemini_{hash(api_key) % 10000}"
        
        try:
            # Create and test engine
            engine = GeminiAnalysisEngine(config)
            await self._test_engine_connection(engine)
            
            # Store engine and mark as healthy
            self.engines[engine_id] = engine
            self.connection_health[engine_id] = {
                'status': 'healthy',
                'last_check': datetime.now().isoformat(),
                'consecutive_failures': 0
            }
            
            logging.info(f"Gemini engine {engine_id} created and validated successfully")
            return engine_id
            
        except Exception as e:
            logging.error(f"Failed to create Gemini engine: {e}")
            raise
    
    async def _test_engine_connection(self, engine: GeminiAnalysisEngine):
        """Test engine connection with minimal request"""
        
        test_request = AnalysisRequest(
            content="# Test Document\n\nThis is a test.",
            analysis_type=AnalysisType.CONTENT_SUMMARY,
            model=GeminiModel.FLASH
        )
        
        response = await engine.analyze_content(test_request)
        if not response.success:
            raise RuntimeError(f"Engine connection test failed: {response.error_message}")
    
    def get_engine(self, engine_id: str) -> Optional[GeminiAnalysisEngine]:
        """Get engine instance by ID"""
        return self.engines.get(engine_id)
    
    def list_engines(self) -> Dict[str, JSONDict]:
        """List all available engines with health status"""
        
        result = {}
        for engine_id, engine in self.engines.items():
            health = self.connection_health.get(engine_id, {})
            metrics = engine.get_performance_metrics()
            
            result[engine_id] = {
                'health_status': health,
                'performance_metrics': metrics,
                'config': {
                    'default_model': engine.config.default_model.value,
                    'max_tokens': engine.config.max_tokens,
                    'temperature': engine.config.temperature
                }
            }
        
        return result
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all engines"""
        
        health_results = {}
        
        for engine_id, engine in self.engines.items():
            try:
                await self._test_engine_connection(engine)
                self.connection_health[engine_id].update({
                    'status': 'healthy',
                    'last_check': datetime.now().isoformat(),
                    'consecutive_failures': 0
                })
                health_results[engine_id] = True
                
            except Exception as e:
                self.connection_health[engine_id]['consecutive_failures'] += 1
                self.connection_health[engine_id]['status'] = 'unhealthy'
                self.connection_health[engine_id]['last_error'] = str(e)
                health_results[engine_id] = False
                
                logging.warning(f"Health check failed for engine {engine_id}: {e}")
        
        return health_results


# Utility Functions for External Integration
def create_analysis_request(
    content: str,
    analysis_type: str,
    model: str = GeminiModel.PRO.value,
    custom_instructions: Optional[str] = None
) -> AnalysisRequest:
    """Factory function for creating analysis requests"""
    
    return AnalysisRequest(
        content=content,
        analysis_type=AnalysisType(analysis_type),
        model=GeminiModel.from_str(model),
        custom_instructions=custom_instructions
    )


def extract_key_insights(analysis_response: AnalysisResponse) -> JSONDict:
    """Extract key insights from analysis response for UI display"""
    
    if not analysis_response.success:
        return {
            'error': True,
            'message': analysis_response.error_message,
            'analysis_type': analysis_response.analysis_type.value
        }
    
    content = analysis_response.content
    insights = {
        'analysis_type': analysis_response.analysis_type.value,
        'model_used': analysis_response.model_used.value,
        'processing_time': analysis_response.processing_time,
        'success': True
    }
    
    # Extract type-specific insights
    if analysis_response.analysis_type == AnalysisType.QUALITY_ANALYSIS:
        insights.update({
            'overall_score': content.get('overall_score', 0),
            'key_scores': {
                'structure': content.get('structure_score', 0),
                'completeness': content.get('completeness_score', 0),
                'accuracy': content.get('accuracy_score', 0),
                'readability': content.get('readability_score', 0)
            },
            'summary': content.get('detailed_feedback', '')[:200] + '...' if content.get('detailed_feedback', '') else ''
        })
    
    elif analysis_response.analysis_type == AnalysisType.CONTENT_SUMMARY:
        insights.update({
            'summary': content.get('executive_summary', ''),
            'topics': content.get('main_topics', []),
            'quality_score': content.get('content_quality', 0)
        })
    
    return insights
JSONDict = Dict[str, JsonValue]
