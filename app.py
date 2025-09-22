# У файлі app.py - замініть імпорти
"""Entry point for the MarkItDown Testing Platform - STRATEGIC ARCHITECTURE."""

from app_interface import (
    create_strategic_hf_spaces_application,    # Нова функція
    create_gradio_app,
    main,
)

# Додайте нові імпорти
from service_layer import PlatformServiceLayer
from event_system import EventOrchestrator
from response_factory import StrategicResponseFactory

__all__ = [
    "create_strategic_hf_spaces_application",
    "create_gradio_app", 
    "main",
    "PlatformServiceLayer",
    "EventOrchestrator", 
    "StrategicResponseFactory",
]

if __name__ == "__main__":
    main()