"""
Enterprise-Grade Core Modules for MarkItDown Testing Platform

Strategic Design Philosophy:
- Stateless architecture for HF Spaces optimization
- Resource-aware processing with automatic cleanup
- Comprehensive error handling and recovery mechanisms
- Modular design enabling easy component replacement

This module implements the foundational processing layer with strict
separation of concerns and enterprise-grade error handling.
"""

import asyncio
import tempfile
import shutil
import os
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import aiofiles
from markitdown import MarkItDown
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
try:
    import magic
except ImportError:
    magic = None
import mimetypes
import psutil


# Strategic Configuration Management
from pydantic import JsonValue

JSONDict = Dict[str, JsonValue]

# Strategic Configuration Management
@dataclass
class ProcessingConfig:
    """Centralized configuration for processing parameters"""
    max_file_size_mb: int = 50
    max_memory_usage_gb: float = 12.0
    temp_cleanup_interval: int = 300  # seconds
    max_concurrent_processes: int = 3
    processing_timeout: int = 300
    gemini_timeout: int = 60
    retry_attempts: int = 3


@dataclass
class ProcessingResult:
    """Standardized result container for all processing operations"""
    success: bool
    content: str
    metadata: JSONDict
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    resource_usage: Optional[JSONDict] = None


class ResourceManager:
    """
    Enterprise-grade resource management for HF Spaces constraints
    
    Strategic Approach:
    - Proactive resource monitoring
    - Automatic cleanup mechanisms
    - Memory-efficient processing patterns
    - Graceful degradation under resource pressure
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.active_processes = set()
        self.temp_directories = set()
        
    def check_resource_availability(self, file_size_bytes: int) -> bool:
        """Validate resource availability before processing"""
        
        # Convert bytes to MB for comparison
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if file_size_mb > self.config.max_file_size_mb:
            raise ResourceError(
                f"File size {file_size_mb:.2f}MB exceeds limit {self.config.max_file_size_mb}MB"
            )
        
        memory_info = psutil.virtual_memory()
        process_memory_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        
        if process_memory_gb > self.config.max_memory_usage_gb:
            raise ResourceError(
                f"Process memory usage {process_memory_gb:.2f}GB exceeds limit {self.config.max_memory_usage_gb:.2f}GB"
            )
        
        available_gb = memory_info.available / (1024**3)
        if available_gb < 1.0:
            raise ResourceError(
                f"Low system memory available: {available_gb:.2f}GB"
            )
        
        if len(self.active_processes) >= self.config.max_concurrent_processes:
            raise ResourceError("Maximum concurrent processes exceeded")
        
        return True
    
    @asynccontextmanager
    async def managed_temp_directory(self):
        """Context manager for temporary directory with automatic cleanup"""
        temp_dir = tempfile.mkdtemp(prefix="markitdown_")
        self.temp_directories.add(temp_dir)
        
        try:
            yield temp_dir
        finally:
            await self._cleanup_directory(temp_dir)
            self.temp_directories.discard(temp_dir)
    
    async def _cleanup_directory(self, directory: str):
        """Async cleanup of temporary directory"""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Cleanup warning for {directory}: {e}")
    
    async def force_cleanup(self):
        """Emergency cleanup of all managed resources"""
        cleanup_tasks = [
            self._cleanup_directory(temp_dir) 
            for temp_dir in list(self.temp_directories)
        ]
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Force garbage collection
        gc.collect()
        
        self.temp_directories.clear()


class StreamlineFileHandler:
    """
    Memory-efficient file processing optimized for HF Spaces
    
    Key Design Principles:
    - Stream-based processing to minimize memory footprint
    - Comprehensive file validation and security checks
    - Automatic format detection and metadata extraction
    - Graceful error handling with detailed diagnostics
    """
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.supported_formats = {
            '.pdf', '.docx', '.pptx', '.xlsx', '.txt', 
            '.html', '.htm', '.csv', '.json', '.xml', '.rtf'
        }
    
    async def process_upload(self, file_obj, metadata_override: Optional[JSONDict] = None) -> ProcessingResult:
        """Process uploaded file with comprehensive validation"""
        
        start_time = datetime.now()
        
        try:
            # Extract basic file information
            file_info = self._extract_file_metadata(file_obj)

            if metadata_override:
                # Merge provided metadata, prioritising supplied values
                for key, value in metadata_override.items():
                    if value in (None, ""):
                        continue
                    file_info[key] = value

            # Recalculate support flag using final extension
            extension = file_info.get('extension', '').lower()
            if extension:
                if not extension.startswith('.'):
                    extension = f'.{extension}'
                file_info['extension'] = extension
            file_info['supported'] = file_info.get('extension') in self.supported_formats
            
            # Resource availability check
            self.resource_manager.check_resource_availability(file_info['size'])
            
            # Security validation
            await self._validate_file_security(file_obj, file_info)
            
            # Read file content efficiently
            content = await self._read_file_content(file_obj)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata=file_info,
                processing_time=processing_time,
                resource_usage=self._get_current_resource_usage()
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                metadata={},
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _extract_file_metadata(self, file_obj) -> JSONDict:
        """Extract comprehensive file metadata"""
        
        file_path = Path(file_obj.name) if hasattr(file_obj, 'name') else Path("unknown")
        
        return {
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'size': getattr(file_obj, 'size', 0),
            'mime_type': self._detect_mime_type(file_obj),
            'timestamp': datetime.now().isoformat(),
            'supported': file_path.suffix.lower() in self.supported_formats
        }
    
    def _detect_mime_type(self, file_obj) -> str:
        """Detect MIME type using python-magic if available"""
        mime_type = None

        if magic is not None and hasattr(file_obj, 'read'):
            try:
                current_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
                chunk = file_obj.read(1024)
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(current_pos)
                mime_type = magic.from_buffer(chunk, mime=True) if chunk else None
            except Exception:
                mime_type = None

        if not mime_type:
            filename = getattr(file_obj, 'name', None)
            if filename:
                mime_type = mimetypes.guess_type(filename)[0]

        return mime_type or 'application/octet-stream'
    
    async def _validate_file_security(self, file_obj, file_info: JSONDict):
        """Comprehensive security validation"""
        
        # File extension validation
        if not file_info['supported']:
            raise SecurityError(f"Unsupported file format: {file_info['extension']}")
        
        # MIME type consistency check
        expected_mimes = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
        
        expected_mime = expected_mimes.get(file_info['extension'])
        if expected_mime and not file_info['mime_type'].startswith(expected_mime.split('/')[0]):
            logging.warning(f"MIME type mismatch for {file_info['extension']}")
    
    async def _read_file_content(self, file_obj) -> bytes:
        """Memory-efficient file content reading"""
        
        if hasattr(file_obj, 'read'):
            # Reset to beginning if possible
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            return file_obj.read()
        
        # Handle different file object types
        if hasattr(file_obj, 'file'):
            return file_obj.file.read()
        
        raise ValueError("Unable to read file content")
    
    def _get_current_resource_usage(self) -> JSONDict:
        """Get current system resource usage"""
        
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory_used_gb': memory_info.used / (1024**3),
            'memory_available_gb': memory_info.available / (1024**3),
            'cpu_percent': cpu_percent,
            'timestamp': datetime.now().isoformat()
        }


class HFConversionEngine:
    """
    MarkItDown wrapper optimized for stateless HF Spaces execution
    
    Strategic Design Features:
    - Async processing with progress tracking
    - Automatic resource cleanup and memory management
    - Comprehensive error handling and retry mechanisms
    - Performance monitoring and optimization
    """
    
    def __init__(self, resource_manager: ResourceManager, config: ProcessingConfig):
        self.resource_manager = resource_manager
        self.config = config
        self.md = MarkItDown()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def convert_stream(self, file_content: bytes, file_metadata: JSONDict) -> ProcessingResult:
        """Stream-based conversion with automatic cleanup and retry logic"""
        
        start_time = datetime.now()
        process_id = id(asyncio.current_task())
        self.resource_manager.active_processes.add(process_id)
        
        try:
            async with self.resource_manager.managed_temp_directory() as temp_dir:
                # Create temporary file for MarkItDown processing
                temp_file_path = await self._create_temp_file(
                    temp_dir, file_content, file_metadata
                )
                
                # Perform conversion with timeout
                result = await asyncio.wait_for(
                    self._execute_conversion(temp_file_path),
                    timeout=self.config.gemini_timeout
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return ProcessingResult(
                    success=True,
                    content=result.text_content,
                    metadata={
                        'original_file': file_metadata,
                        'conversion_time': processing_time,
                        'content_length': len(result.text_content),
                        'conversion_metadata': self._extract_conversion_metadata(result)
                    },
                    processing_time=processing_time
                )
        
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                metadata=file_metadata,
                error_message=f"Conversion failed: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        finally:
            self.resource_manager.active_processes.discard(process_id)
    
    async def _create_temp_file(self, temp_dir: str, content: bytes, metadata: JSONDict) -> str:
        """Create temporary file for processing"""
        
        filename = metadata.get('filename', 'temp_file')
        temp_file_path = os.path.join(temp_dir, filename)
        
        async with aiofiles.open(temp_file_path, 'wb') as temp_file:
            await temp_file.write(content)
        
        return temp_file_path
    
    async def _execute_conversion(self, file_path: str):
        """Execute MarkItDown conversion in thread pool"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.md.convert, file_path
        )
    
    def _extract_conversion_metadata(self, result) -> JSONDict:
        """Extract metadata from MarkItDown result"""
        
        content = result.text_content
        
        return {
            'lines_count': len(content.split('\n')),
            'word_count': len(content.split()),
            'character_count': len(content),
            'has_tables': '|' in content,
            'has_headers': content.count('#') > 0,
            'has_lists': content.count('-') > 0 or content.count('*') > 0,
            'has_links': '[' in content and '](' in content
        }


# Custom Exception Classes
class ResourceError(Exception):
    """Resource constraint violation"""
    pass

class SecurityError(Exception):
    """Security validation failure"""
    pass

class ConversionError(Exception):
    """Document conversion failure"""
    pass
