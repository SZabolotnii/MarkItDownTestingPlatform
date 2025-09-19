"""Entry point for the MarkItDown Testing Platform."""

from app_interface import (
    ApplicationFactory,
    MarkItDownTestingApp,
    create_gradio_app,
    main,
)
from app_logic import (
    DocumentProcessingOrchestrator,
    ProcessingRequest,
    ProcessingResponse,
)

__all__ = [
    "MarkItDownTestingApp",
    "ApplicationFactory",
    "DocumentProcessingOrchestrator",
    "ProcessingRequest",
    "ProcessingResponse",
    "create_gradio_app",
    "main",
]


if __name__ == "__main__":
    main()
