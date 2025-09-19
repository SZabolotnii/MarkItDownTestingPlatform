"""
MarkItDown Testing Platform - Usage Examples and Testing Suite

This module provides comprehensive examples and testing capabilities for the
MarkItDown Testing Platform, demonstrating various use cases and validation scenarios.

Strategic Examples Coverage:
- Basic document conversion workflows
- Advanced AI analysis integration
- Performance benchmarking and optimization
- Enterprise integration patterns
- Error handling and recovery scenarios
"""

import asyncio
import tempfile
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import platform components
from core.modules import (
    StreamlineFileHandler, HFConversionEngine, ResourceManager,
    ProcessingConfig, ProcessingResult
)
from llm.gemini_connector import (
    GeminiAnalysisEngine, GeminiConfig, AnalysisRequest,
    AnalysisType, GeminiModel, create_analysis_request
)
from visualization.analytics_engine import (
    InteractiveVisualizationEngine, QualityMetricsCalculator,
    VisualizationConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentSampleGenerator:
    """Generate test documents for comprehensive platform testing"""
    
    @staticmethod
    def create_test_html() -> str:
        """Create comprehensive HTML test document"""
        
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MarkItDown Test Document</title>
            <style>
                .highlight { background-color: yellow; }
                .important { font-weight: bold; color: red; }
            </style>
        </head>
        <body>
            <h1>Enterprise Document Conversion Test</h1>
            <p class="important">This is a comprehensive test document for MarkItDown platform validation.</p>
            
            <h2>Document Structure Testing</h2>
            <p>This section tests various structural elements and their conversion accuracy.</p>
            
            <h3>List Testing</h3>
            <h4>Unordered Lists</h4>
            <ul>
                <li>Primary list item with <strong>bold text</strong></li>
                <li>Secondary item with <em>italic formatting</em></li>
                <li>Nested list testing:
                    <ul>
                        <li>Nested item 1</li>
                        <li>Nested item 2 with <a href="https://example.com">external link</a></li>
                    </ul>
                </li>
                <li>Code reference: <code>function processDocument()</code></li>
            </ul>
            
            <h4>Ordered Lists</h4>
            <ol>
                <li>First priority task</li>
                <li>Second priority with emphasis: <span class="highlight">critical deadline</span></li>
                <li>Third priority item</li>
            </ol>
            
            <h3>Table Structure Testing</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th>Feature</th>
                        <th>Status</th>
                        <th>Priority</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Document Conversion</td>
                        <td>✅ Complete</td>
                        <td>High</td>
                        <td>Core functionality working</td>
                    </tr>
                    <tr>
                        <td>AI Analysis</td>
                        <td>🔄 In Progress</td>
                        <td>High</td>
                        <td>Gemini integration active</td>
                    </tr>
                    <tr>
                        <td>Visualization</td>
                        <td>✅ Complete</td>
                        <td>Medium</td>
                        <td>Interactive dashboards ready</td>
                    </tr>
                    <tr>
                        <td>Export Features</td>
                        <td>⏳ Planned</td>
                        <td>Low</td>
                        <td>Multiple format support</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>Code Block Testing</h3>
            <p>Example Python integration code:</p>
            <pre><code>
from markitdown import MarkItDown
from gemini_connector import GeminiAnalysisEngine

async def process_document(file_path, api_key):
    # Initialize components
    md = MarkItDown()
    gemini = GeminiAnalysisEngine(api_key)
    
    # Convert document
    result = md.convert(file_path)
    
    # Analyze with AI
    analysis = await gemini.analyze_content(result.text_content)
    
    return result, analysis
            </code></pre>
            
            <h3>Link and Reference Testing</h3>
            <p>This section contains various types of references:</p>
            <ul>
                <li>External link: <a href="https://github.com/microsoft/markitdown">Microsoft MarkItDown Repository</a></li>
                <li>Email reference: <a href="mailto:support@example.com">Technical Support</a></li>
                <li>Internal reference: <a href="#document-structure-testing">Jump to Structure Section</a></li>
                <li>Document reference: See the <a href="./documentation.pdf">full documentation</a> for details</li>
            </ul>
            
            <h3>Special Formatting Testing</h3>
            <div>
                <p><strong>Bold text emphasis</strong> and <em>italic styling</em> combined with <u>underlined content</u>.</p>
                <p><del>Strikethrough text</del> and <mark>highlighted content</mark> for attention.</p>
                <p>Mathematical notation: E = mc<sup>2</sup> and chemical formula: H<sub>2</sub>O.</p>
            </div>
            
            <h2>Content Quality Assessment</h2>
            <blockquote style="border-left: 4px solid #ccc; padding-left: 16px; font-style: italic;">
                "Quality is not an act, it is a habit. The systematic approach to document conversion 
                and analysis ensures consistent, reliable results across diverse content types and formats."
            </blockquote>
            
            <h3>Technical Specifications</h3>
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                <h4>Processing Requirements:</h4>
                <ul>
                    <li><strong>Maximum File Size:</strong> 50MB (HF Spaces limit)</li>
                    <li><strong>Supported Formats:</strong> PDF, DOCX, PPTX, XLSX, HTML, TXT, CSV, JSON, XML</li>
                    <li><strong>Processing Timeout:</strong> 5 minutes maximum</li>
                    <li><strong>Memory Usage:</strong> Optimized for 16GB constraint</li>
                </ul>
            </div>
            
            <h2>Integration Examples</h2>
            <p>The following examples demonstrate enterprise integration patterns:</p>
            
            <h3>Batch Processing Workflow</h3>
            <ol>
                <li>Document ingestion from multiple sources</li>
                <li>Automated quality validation pipeline</li>
                <li>AI-powered content analysis and enhancement</li>
                <li>Structured output generation for downstream systems</li>
                <li>Comprehensive reporting and analytics</li>
            </ol>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
                <p><em>Generated for MarkItDown Testing Platform - Version 1.0.0</em></p>
                <p><strong>Document ID:</strong> TEST-DOC-001 | <strong>Created:</strong> {timestamp}</p>
            </footer>
        </body>
        </html>
        """.replace('{timestamp}', datetime.now().isoformat())
    
    @staticmethod
    def create_test_json() -> str:
        """Create structured JSON test data"""
        
        return json.dumps({
            "document_metadata": {
                "title": "MarkItDown Test Configuration",
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "description": "Comprehensive test data for platform validation"
            },
            "processing_config": {
                "max_file_size_mb": 50,
                "timeout_seconds": 300,
                "supported_formats": [
                    "pdf", "docx", "pptx", "xlsx", 
                    "html", "txt", "csv", "json", "xml"
                ],
                "ai_analysis": {
                    "enabled": True,
                    "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
                    "analysis_types": [
                        "quality_analysis",
                        "structure_review", 
                        "content_summary",
                        "extraction_quality"
                    ]
                }
            },
            "test_scenarios": [
                {
                    "name": "Basic Document Conversion",
                    "description": "Test core MarkItDown functionality",
                    "expected_elements": [
                        "headers", "paragraphs", "lists", "tables", "links"
                    ],
                    "quality_threshold": 7.0
                },
                {
                    "name": "AI Analysis Integration",
                    "description": "Test Gemini API integration",
                    "required_api_key": True,
                    "expected_analysis": [
                        "overall_score", "detailed_feedback", "recommendations"
                    ],
                    "quality_threshold": 8.0
                },
                {
                    "name": "Performance Benchmarking",
                    "description": "Test processing speed and resource usage",
                    "metrics": [
                        "processing_time", "memory_usage", "cpu_utilization"
                    ],
                    "performance_threshold": {
                        "processing_time_seconds": 60,
                        "memory_usage_mb": 1000
                    }
                }
            ],
            "quality_metrics": {
                "structural_integrity": {
                    "weight": 0.3,
                    "components": ["headers", "lists", "tables", "formatting"]
                },
                "content_preservation": {
                    "weight": 0.25,
                    "components": ["text_accuracy", "link_preservation", "data_integrity"]
                },
                "ai_analysis_quality": {
                    "weight": 0.25,
                    "components": ["insight_depth", "recommendation_quality", "accuracy"]
                },
                "processing_efficiency": {
                    "weight": 0.2,
                    "components": ["speed", "resource_usage", "reliability"]
                }
            },
            "expected_outputs": {
                "markdown_conversion": {
                    "min_length": 1000,
                    "required_elements": ["# ", "## ", "- ", "| "],
                    "quality_indicators": ["proper_escaping", "structure_preservation"]
                },
                "ai_analysis": {
                    "required_fields": ["overall_score", "detailed_feedback"],
                    "score_range": [0, 10],
                    "feedback_min_length": 100
                },
                "visualization": {
                    "chart_types": ["radar", "bar", "treemap", "line"],
                    "interactive_elements": True,
                    "export_formats": ["html", "png", "svg"]
                }
            }
        }, indent=2)
    
    @staticmethod
    def create_test_csv() -> str:
        """Create CSV test data with various data types"""
        
        return """Name,Age,Department,Salary,Join Date,Performance Rating,Notes
John Smith,34,Engineering,75000,2023-01-15,4.5,"Excellent problem solver, team lead"
Maria Garcia,28,Marketing,62000,2023-03-20,4.2,"Creative campaigns, social media expert"  
David Chen,41,Finance,82000,2022-08-10,4.8,"CPA certified, process optimization"
Sarah Johnson,29,Engineering,68000,2023-02-28,4.3,"Full-stack developer, agile advocate"
Michael Brown,36,Sales,71000,2022-11-05,4.6,"Top performer, client relationship expert"
Lisa Wang,32,Product,78000,2023-01-08,4.4,"UX specialist, user research focused"
Robert Davis,45,Operations,69000,2022-07-22,4.1,"Supply chain optimization, vendor management"
Jennifer Wilson,33,HR,59000,2023-04-12,4.3,"Talent acquisition, employee engagement"
James Anderson,38,Engineering,81000,2022-09-18,4.7,"Senior architect, technical mentoring"
Emily Taylor,27,Marketing,57000,2023-05-01,4.0,"Digital marketing, content strategy"
"""


class PlatformTester:
    """Comprehensive testing suite for the MarkItDown Testing Platform"""
    
    def __init__(self):
        # Initialize platform components
        self.processing_config = ProcessingConfig()
        self.resource_manager = ResourceManager(self.processing_config)
        self.file_handler = StreamlineFileHandler(self.resource_manager)
        self.conversion_engine = HFConversionEngine(self.resource_manager, self.processing_config)
        self.viz_engine = InteractiveVisualizationEngine()
        self.quality_calculator = QualityMetricsCalculator()
        
        # Test results storage
        self.test_results = []
        self.performance_metrics = []
    
    async def run_basic_conversion_test(self) -> Dict[str, Any]:
        """Test basic document conversion functionality"""
        
        logger.info("Running basic conversion test...")
        
        test_start = time.time()
        
        try:
            # Create test HTML document
            html_content = DocumentSampleGenerator.create_test_html()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name
            
            # Simulate file upload
            class MockFile:
                def __init__(self, path):
                    self.name = path
                    with open(path, 'rb') as f:
                        self.content = f.read()
                    self.size = len(self.content)
                
                def read(self):
                    return self.content
            
            mock_file = MockFile(temp_file_path)
            
            # Process file
            file_result = await self.file_handler.process_upload(mock_file)
            
            if not file_result.success:
                return {
                    'test_name': 'basic_conversion',
                    'status': 'failed',
                    'error': file_result.error_message,
                    'duration': time.time() - test_start
                }
            
            # Convert document
            conversion_result = await self.conversion_engine.convert_stream(
                mock_file.content, file_result.metadata
            )
            
            if not conversion_result.success:
                return {
                    'test_name': 'basic_conversion',
                    'status': 'failed', 
                    'error': conversion_result.error_message,
                    'duration': time.time() - test_start
                }
            
            # Validate conversion results
            validation_results = self._validate_conversion_output(conversion_result)
            
            # Calculate quality metrics
            quality_metrics = self.quality_calculator.calculate_conversion_quality_metrics(
                conversion_result
            )
            
            test_duration = time.time() - test_start
            
            # Clean up
            Path(temp_file_path).unlink(missing_ok=True)
            
            return {
                'test_name': 'basic_conversion',
                'status': 'passed',
                'duration': test_duration,
                'validation': validation_results,
                'quality_metrics': quality_metrics,
                'performance': {
                    'processing_time': conversion_result.processing_time,
                    'content_length': len(conversion_result.content),
                    'throughput': len(conversion_result.content) / test_duration
                }
            }
            
        except Exception as e:
            return {
                'test_name': 'basic_conversion',
                'status': 'error',
                'error': str(e),
                'duration': time.time() - test_start
            }
    
    async def run_ai_analysis_test(self, gemini_api_key: str) -> Dict[str, Any]:
        """Test AI analysis integration with Gemini"""
        
        logger.info("Running AI analysis test...")
        
        if not gemini_api_key:
            return {
                'test_name': 'ai_analysis',
                'status': 'skipped',
                'reason': 'No API key provided'
            }
        
        test_start = time.time()
        
        try:
            # Create Gemini engine
            gemini_config = GeminiConfig(api_key=gemini_api_key)
            gemini_engine = GeminiAnalysisEngine(gemini_config)
            
            # Create test content
            test_content = """
            # Test Document for AI Analysis
            
            This is a comprehensive test document designed to evaluate the AI analysis capabilities
            of the MarkItDown Testing Platform.
            
            ## Document Structure
            
            ### Headers and Organization
            This document contains multiple heading levels to test structure recognition.
            
            ### Content Quality
            The content includes various elements:
            - Technical terminology and concepts
            - Business-oriented language and metrics
            - Complex sentence structures
            - Tables and structured data
            
            | Metric | Value | Status |
            |--------|-------|--------|
            | Conversion Quality | 8.5/10 | Excellent |
            | Processing Speed | 2.3s | Good |
            | Resource Usage | 45% | Optimal |
            
            ## Analysis Requirements
            
            This content should trigger comprehensive analysis covering:
            1. **Structure Assessment**: Header hierarchy and organization
            2. **Content Quality**: Information density and clarity
            3. **Technical Accuracy**: Preservation of data and formatting
            4. **Readability**: AI-friendly output optimization
            
            The analysis should provide actionable insights and recommendations
            for improving document conversion processes.
            """
            
            # Test different analysis types
            analysis_types = [
                AnalysisType.QUALITY_ANALYSIS,
                AnalysisType.STRUCTURE_REVIEW,
                AnalysisType.CONTENT_SUMMARY
            ]
            
            analysis_results = {}
            
            for analysis_type in analysis_types:
                analysis_request = AnalysisRequest(
                    content=test_content,
                    analysis_type=analysis_type,
                    model=GeminiModel.PRO
                )
                
                result = await gemini_engine.analyze_content(analysis_request)
                analysis_results[analysis_type.value] = {
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'content_length': len(str(result.content)) if result.success else 0,
                    'error': result.error_message if not result.success else None
                }
            
            test_duration = time.time() - test_start
            
            # Calculate success rate
            successful_analyses = sum(1 for r in analysis_results.values() if r['success'])
            success_rate = successful_analyses / len(analysis_types) * 100
            
            return {
                'test_name': 'ai_analysis',
                'status': 'passed' if success_rate > 0 else 'failed',
                'duration': test_duration,
                'success_rate': success_rate,
                'analysis_results': analysis_results,
                'performance_metrics': gemini_engine.get_performance_metrics()
            }
            
        except Exception as e:
            return {
                'test_name': 'ai_analysis',
                'status': 'error',
                'error': str(e),
                'duration': time.time() - test_start
            }
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        
        logger.info("Running performance benchmark...")
        
        benchmark_start = time.time()
        benchmark_results = {
            'test_name': 'performance_benchmark',
            'start_time': benchmark_start,
            'scenarios': []
        }
        
        # Test scenarios with different file sizes and types
        test_scenarios = [
            {
                'name': 'Small HTML Document',
                'content': DocumentSampleGenerator.create_test_html()[:1000],
                'format': 'html'
            },
            {
                'name': 'Medium HTML Document', 
                'content': DocumentSampleGenerator.create_test_html(),
                'format': 'html'
            },
            {
                'name': 'Large HTML Document',
                'content': DocumentSampleGenerator.create_test_html() * 3,
                'format': 'html'
            },
            {
                'name': 'Structured JSON Data',
                'content': DocumentSampleGenerator.create_test_json(),
                'format': 'json'
            },
            {
                'name': 'CSV Data Table',
                'content': DocumentSampleGenerator.create_test_csv(),
                'format': 'csv'
            }
        ]
        
        for scenario in test_scenarios:
            scenario_start = time.time()
            
            try:
                # Create temporary file
                suffix = f".{scenario['format']}"
                with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as temp_file:
                    temp_file.write(scenario['content'])
                    temp_file_path = temp_file.name
                
                # Simulate file processing
                class MockFile:
                    def __init__(self, path, content):
                        self.name = path
                        self.content = content.encode('utf-8')
                        self.size = len(self.content)
                    
                    def read(self):
                        return self.content
                
                mock_file = MockFile(temp_file_path, scenario['content'])
                
                # Measure processing steps
                step_timings = {}
                
                # File handling
                step_start = time.time()
                file_result = await self.file_handler.process_upload(mock_file)
                step_timings['file_handling'] = time.time() - step_start
                
                if file_result.success:
                    # Document conversion
                    step_start = time.time()
                    conversion_result = await self.conversion_engine.convert_stream(
                        mock_file.content, file_result.metadata
                    )
                    step_timings['conversion'] = time.time() - step_start
                    
                    if conversion_result.success:
                        # Quality metrics calculation
                        step_start = time.time()
                        quality_metrics = self.quality_calculator.calculate_conversion_quality_metrics(
                            conversion_result
                        )
                        step_timings['quality_calculation'] = time.time() - step_start
                        
                        scenario_duration = time.time() - scenario_start
                        
                        scenario_result = {
                            'name': scenario['name'],
                            'status': 'success',
                            'duration': scenario_duration,
                            'step_timings': step_timings,
                            'content_stats': {
                                'input_size': len(scenario['content']),
                                'output_size': len(conversion_result.content),
                                'compression_ratio': len(conversion_result.content) / len(scenario['content'])
                            },
                            'performance_metrics': {
                                'throughput_chars_per_sec': len(scenario['content']) / scenario_duration,
                                'processing_efficiency': quality_metrics.get('composite_score', 0) / scenario_duration
                            }
                        }
                    else:
                        scenario_result = {
                            'name': scenario['name'],
                            'status': 'conversion_failed',
                            'error': conversion_result.error_message,
                            'duration': time.time() - scenario_start
                        }
                else:
                    scenario_result = {
                        'name': scenario['name'],
                        'status': 'file_handling_failed',
                        'error': file_result.error_message,
                        'duration': time.time() - scenario_start
                    }
                
                benchmark_results['scenarios'].append(scenario_result)
                
                # Clean up
                Path(temp_file_path).unlink(missing_ok=True)
                
            except Exception as e:
                benchmark_results['scenarios'].append({
                    'name': scenario['name'],
                    'status': 'error',
                    'error': str(e),
                    'duration': time.time() - scenario_start
                })
        
        # Calculate overall benchmark metrics
        successful_scenarios = [s for s in benchmark_results['scenarios'] if s['status'] == 'success']
        total_duration = time.time() - benchmark_start
        
        benchmark_results.update({
            'total_duration': total_duration,
            'scenarios_total': len(test_scenarios),
            'scenarios_successful': len(successful_scenarios),
            'success_rate': len(successful_scenarios) / len(test_scenarios) * 100,
            'average_processing_time': sum(s['duration'] for s in successful_scenarios) / len(successful_scenarios) if successful_scenarios else 0,
            'total_throughput': sum(s.get('performance_metrics', {}).get('throughput_chars_per_sec', 0) for s in successful_scenarios),
            'status': 'passed' if len(successful_scenarios) > len(test_scenarios) / 2 else 'failed'
        })
        
        return benchmark_results
    
    async def run_visualization_test(self) -> Dict[str, Any]:
        """Test visualization generation capabilities"""
        
        logger.info("Running visualization test...")
        
        test_start = time.time()
        
        try:
            # Create mock conversion result for testing
            mock_conversion_result = ProcessingResult(
                success=True,
                content=DocumentSampleGenerator.create_test_html(),
                metadata={
                    'original_file': {
                        'filename': 'test_document.html',
                        'size': 5000,
                        'extension': '.html'
                    }
                },
                processing_time=2.5
            )
            
            # Test visualization generation
            visualization_tests = []
            
            # Quality Dashboard Test
            try:
                dashboard_start = time.time()
                quality_dashboard = self.viz_engine.create_quality_dashboard(mock_conversion_result)
                dashboard_duration = time.time() - dashboard_start
                
                visualization_tests.append({
                    'name': 'quality_dashboard',
                    'status': 'success',
                    'duration': dashboard_duration,
                    'chart_type': 'multi-chart dashboard',
                    'data_points': len(quality_dashboard.data) if hasattr(quality_dashboard, 'data') else 'multiple'
                })
            except Exception as e:
                visualization_tests.append({
                    'name': 'quality_dashboard',
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Structure Analysis Test
            try:
                structure_start = time.time()
                structure_viz = self.viz_engine.create_structural_analysis_viz(mock_conversion_result)
                structure_duration = time.time() - structure_start
                
                visualization_tests.append({
                    'name': 'structure_analysis',
                    'status': 'success', 
                    'duration': structure_duration,
                    'chart_type': 'structural analysis',
                    'components': 'treemap, pie, bar, scatter'
                })
            except Exception as e:
                visualization_tests.append({
                    'name': 'structure_analysis',
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Export Ready Report Test
            try:
                report_start = time.time()
                export_report = self.viz_engine.create_export_ready_report(mock_conversion_result)
                report_duration = time.time() - report_start
                
                visualization_tests.append({
                    'name': 'export_report',
                    'status': 'success',
                    'duration': report_duration,
                    'chart_count': len(export_report),
                    'report_types': list(export_report.keys())
                })
            except Exception as e:
                visualization_tests.append({
                    'name': 'export_report',
                    'status': 'failed',
                    'error': str(e)
                })
            
            test_duration = time.time() - test_start
            successful_tests = [t for t in visualization_tests if t['status'] == 'success']
            
            return {
                'test_name': 'visualization',
                'status': 'passed' if len(successful_tests) > 0 else 'failed',
                'duration': test_duration,
                'tests_run': len(visualization_tests),
                'tests_successful': len(successful_tests),
                'success_rate': len(successful_tests) / len(visualization_tests) * 100,
                'test_details': visualization_tests
            }
            
        except Exception as e:
            return {
                'test_name': 'visualization',
                'status': 'error',
                'error': str(e),
                'duration': time.time() - test_start
            }
    
    async def run_comprehensive_test_suite(self, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Run complete test suite with all components"""
        
        logger.info("Starting comprehensive test suite...")
        
        suite_start = time.time()
        
        # Run all tests
        test_results = []
        
        # Basic conversion test
        basic_test = await self.run_basic_conversion_test()
        test_results.append(basic_test)
        
        # AI analysis test (if API key provided)
        if gemini_api_key:
            ai_test = await self.run_ai_analysis_test(gemini_api_key)
            test_results.append(ai_test)
        
        # Performance benchmark
        perf_test = await self.run_performance_benchmark()
        test_results.append(perf_test)
        
        # Visualization test
        viz_test = await self.run_visualization_test()
        test_results.append(viz_test)
        
        # Calculate overall results
        suite_duration = time.time() - suite_start
        passed_tests = [t for t in test_results if t.get('status') == 'passed']
        failed_tests = [t for t in test_results if t.get('status') in ['failed', 'error']]
        
        # Generate comprehensive report
        comprehensive_report = {
            'test_suite': 'MarkItDown Platform Comprehensive Test',
            'timestamp': datetime.now().isoformat(),
            'duration': suite_duration,
            'summary': {
                'total_tests': len(test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'skipped': len([t for t in test_results if t.get('status') == 'skipped']),
                'success_rate': len(passed_tests) / len(test_results) * 100
            },
            'test_results': test_results,
            'system_info': self._get_system_info(),
            'recommendations': self._generate_recommendations(test_results),
            'overall_status': 'PASSED' if len(passed_tests) > len(test_results) / 2 else 'FAILED'
        }
        
        return comprehensive_report
    
    def _validate_conversion_output(self, conversion_result: ProcessingResult) -> Dict[str, Any]:
        """Validate conversion output quality and completeness"""
        
        content = conversion_result.content
        validation_results = {
            'content_length_ok': len(content) > 100,
            'has_headers': content.count('#') > 0,
            'has_lists': content.count('- ') > 0 or content.count('* ') > 0,
            'has_tables': content.count('|') > 0,
            'has_links': content.count('](') > 0,
            'proper_encoding': all(ord(char) < 128 for char in content[:1000]),  # ASCII check sample
            'no_empty_sections': not bool(content.count('##\n\n##'))
        }
        
        # Calculate validation score
        validation_score = sum(validation_results.values()) / len(validation_results)
        validation_results['overall_score'] = validation_score
        validation_results['status'] = 'passed' if validation_score > 0.7 else 'warning' if validation_score > 0.5 else 'failed'
        
        return validation_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test report"""
        
        try:
            import psutil
            import platform
            
            memory = psutil.virtual_memory()
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'architecture': platform.architecture()[0]
            }
        except Exception as e:
            return {'error': f'Could not gather system info: {e}'}
    
    def _generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Analyze test results for recommendations
        for test in test_results:
            if test.get('status') == 'failed':
                test_name = test.get('test_name', 'unknown')
                recommendations.append(f"❌ {test_name.title()} test failed - investigate {test.get('error', 'unknown error')}")
            
            elif test.get('status') == 'passed':
                test_name = test.get('test_name', 'unknown')
                
                # Performance recommendations
                if 'duration' in test and test['duration'] > 30:
                    recommendations.append(f"⚠️ {test_name.title()} test took {test['duration']:.2f}s - consider optimization")
                
                # Success rate recommendations
                if 'success_rate' in test and test['success_rate'] < 90:
                    recommendations.append(f"⚠️ {test_name.title()} success rate is {test['success_rate']:.1f}% - investigate reliability issues")
        
        # General recommendations
        if not any('ai_analysis' in str(test) for test in test_results):
            recommendations.append("💡 Consider adding Gemini API key for AI analysis testing")
        
        if not recommendations:
            recommendations.append("✅ All tests passed successfully - platform ready for production use")
        
        return recommendations


class UsageExamples:
    """Practical usage examples for different scenarios"""
    
    @staticmethod
    async def example_basic_usage():
        """Example: Basic document conversion"""
        
        print("=== Basic Document Conversion Example ===")
        
        # Initialize components
        config = ProcessingConfig()
        resource_manager = ResourceManager(config)
        file_handler = StreamlineFileHandler(resource_manager)
        conversion_engine = HFConversionEngine(resource_manager, config)
        
        # Create sample document
        sample_html = DocumentSampleGenerator.create_test_html()
        
        # Simulate file upload
        class MockFile:
            def __init__(self, content):
                self.name = "sample.html"
                self.content = content.encode('utf-8')
                self.size = len(self.content)
            
            def read(self):
                return self.content
        
        mock_file = MockFile(sample_html)
        
        try:
            # Process file
            print("1. Processing uploaded file...")
            file_result = await file_handler.process_upload(mock_file)
            
            if file_result.success:
                print(f"   ✅ File processed: {file_result.metadata['filename']}")
                
                # Convert document
                print("2. Converting to Markdown...")
                conversion_result = await conversion_engine.convert_stream(
                    mock_file.content, file_result.metadata
                )
                
                if conversion_result.success:
                    print(f"   ✅ Conversion successful in {conversion_result.processing_time:.2f}s")
                    print(f"   📄 Generated {len(conversion_result.content)} characters")
                    print(f"   📋 Preview: {conversion_result.content[:200]}...")
                    
                    # Calculate quality metrics
                    print("3. Calculating quality metrics...")
                    quality_calculator = QualityMetricsCalculator()
                    metrics = quality_calculator.calculate_conversion_quality_metrics(conversion_result)
                    
                    print(f"   📊 Composite Score: {metrics.get('composite_score', 0):.1f}/10")
                    print(f"   📈 Word Count: {metrics.get('basic_metrics', {}).get('total_words', 0)}")
                    print(f"   🏗️ Structure Elements: {metrics.get('structural_metrics', {}).get('header_count', 0)} headers")
                    
                else:
                    print(f"   ❌ Conversion failed: {conversion_result.error_message}")
            else:
                print(f"   ❌ File processing failed: {file_result.error_message}")
                
        except Exception as e:
            print(f"   ❌ Example failed: {e}")
        
        print("\n" + "="*50 + "\n")
    
    @staticmethod
    async def example_ai_integration(api_key: str):
        """Example: AI-powered analysis integration"""
        
        if not api_key:
            print("=== AI Integration Example (Skipped - No API Key) ===\n")
            return
        
        print("=== AI-Powered Analysis Example ===")
        
        try:
            # Initialize Gemini engine
            print("1. Initializing Gemini AI...")
            gemini_config = GeminiConfig(api_key=api_key)
            gemini_engine = GeminiAnalysisEngine(gemini_config)
            
            # Sample content for analysis
            sample_content = """
            # Enterprise Document Management Strategy
            
            ## Executive Summary
            This document outlines our comprehensive approach to modernizing document 
            management processes through automated conversion and AI-powered analysis.
            
            ## Key Objectives
            1. **Standardization**: Convert legacy formats to modern, searchable formats
            2. **Quality Assurance**: Implement AI-driven quality validation
            3. **Efficiency**: Reduce manual processing time by 75%
            4. **Scalability**: Handle 10,000+ documents monthly
            
            ## Implementation Timeline
            
            | Phase | Duration | Deliverables |
            |-------|----------|--------------|
            | Phase 1 | 2 months | Platform deployment |
            | Phase 2 | 3 months | AI integration |
            | Phase 3 | 1 month | Quality validation |
            
            ## Expected ROI
            - Processing time reduction: 75%
            - Quality improvement: 40%
            - Cost savings: $50,000 annually
            """
            
            # Test different analysis types
            analysis_types = [
                (AnalysisType.QUALITY_ANALYSIS, "Quality Assessment"),
                (AnalysisType.CONTENT_SUMMARY, "Content Summary"),
                (AnalysisType.STRUCTURE_REVIEW, "Structure Analysis")
            ]
            
            for analysis_type, description in analysis_types:
                print(f"\n2. Running {description}...")
                
                request = AnalysisRequest(
                    content=sample_content,
                    analysis_type=analysis_type,
                    model=GeminiModel.PRO
                )
                
                result = await gemini_engine.analyze_content(request)
                
                if result.success:
                    print(f"   ✅ {description} completed in {result.processing_time:.2f}s")
                    
                    if analysis_type == AnalysisType.QUALITY_ANALYSIS:
                        content = result.content
                        print(f"   📊 Overall Score: {content.get('overall_score', 0)}/10")
                        print(f"   🏗️ Structure Score: {content.get('structure_score', 0)}/10")
                        print(f"   📋 Completeness: {content.get('completeness_score', 0)}/10")
                        
                    elif analysis_type == AnalysisType.CONTENT_SUMMARY:
                        summary = result.content.get('executive_summary', '')[:200]
                        print(f"   📝 Summary: {summary}...")
                        
                else:
                    print(f"   ❌ {description} failed: {result.error_message}")
            
            # Performance metrics
            print(f"\n3. Performance Metrics:")
            perf_metrics = gemini_engine.get_performance_metrics()
            print(f"   📈 Total Requests: {perf_metrics['total_requests']}")
            print(f"   ⏱️ Average Time: {perf_metrics['average_processing_time']:.2f}s")
            print(f"   ✅ Success Rate: {perf_metrics['success_rate_percent']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ AI Integration example failed: {e}")
        
        print("\n" + "="*50 + "\n")
    
    @staticmethod
    async def example_visualization_generation():
        """Example: Generate interactive visualizations"""
        
        print("=== Visualization Generation Example ===")
        
        try:
            # Create mock results for visualization
            mock_result = ProcessingResult(
                success=True,
                content=DocumentSampleGenerator.create_test_html(),
                metadata={
                    'original_file': {'filename': 'test.html', 'size': 5000}
                },
                processing_time=2.3
            )
            
            # Initialize visualization engine
            print("1. Initializing visualization engine...")
            viz_engine = InteractiveVisualizationEngine()
            
            # Generate quality dashboard
            print("2. Creating quality dashboard...")
            dashboard_start = time.time()
            quality_dashboard = viz_engine.create_quality_dashboard(mock_result)
            dashboard_time = time.time() - dashboard_start
            
            print(f"   ✅ Quality dashboard generated in {dashboard_time:.2f}s")
            print(f"   📊 Chart components: {len(quality_dashboard.data)} data traces")
            
            # Generate structure analysis
            print("3. Creating structure analysis...")
            structure_start = time.time()
            structure_viz = viz_engine.create_structural_analysis_viz(mock_result)
            structure_time = time.time() - structure_start
            
            print(f"   ✅ Structure analysis generated in {structure_time:.2f}s")
            
            # Generate export report
            print("4. Creating export-ready report...")
            report_start = time.time()
            export_report = viz_engine.create_export_ready_report(mock_result)
            report_time = time.time() - report_start
            
            print(f"   ✅ Export report generated in {report_time:.2f}s")
            print(f"   📈 Report components: {list(export_report.keys())}")
            
            total_time = dashboard_time + structure_time + report_time
            print(f"\n   📊 Total visualization time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Visualization example failed: {e}")
        
        print("\n" + "="*50 + "\n")


async def main():
    """Main function to run examples and tests"""
    
    print("🚀 MarkItDown Testing Platform - Examples & Testing Suite")
    print("=" * 60)
    
    # Run usage examples
    await UsageExamples.example_basic_usage()
    
    # Ask for Gemini API key for AI examples
    api_key = input("Enter Gemini API key for AI examples (press Enter to skip): ").strip()
    if api_key:
        await UsageExamples.example_ai_integration(api_key)
    
    await UsageExamples.example_visualization_generation()
    
    # Run comprehensive test suite
    print("🧪 Running Comprehensive Test Suite...")
    print("=" * 40)
    
    tester = PlatformTester()
    test_results = await tester.run_comprehensive_test_suite(api_key if api_key else None)
    
    # Display test results
    print(f"\n📊 Test Suite Results:")
    print(f"   Status: {test_results['overall_status']}")
    print(f"   Duration: {test_results['duration']:.2f}s")
    print(f"   Success Rate: {test_results['summary']['success_rate']:.1f}%")
    print(f"   Tests: {test_results['summary']['passed']}/{test_results['summary']['total_tests']} passed")
    
    if test_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in test_results['recommendations'][:5]:  # Show top 5
            print(f"   {rec}")
    
    # Save detailed results
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print("\n✅ Examples and testing complete!")


if __name__ == "__main__":
    asyncio.run(main())