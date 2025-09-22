# Створіть файл quick_test.py
"""Швидкий тест архітектури"""

import asyncio
import logging
from service_layer import create_hf_spaces_service_layer
from ui_components import UIComponentFactory  
from response_factory import StrategicResponseFactory
from event_system import EventOrchestrator
from app_logic import DocumentProcessingOrchestrator

async def test_architecture():
    print("🧪 Тестування стратегічної архітектури...")
    
    try:
        # Тест 1: Service Layer
        print("1️⃣ Тестування Service Layer...")
        # Створіть мок orchestrator для тестування
        from core.modules import ProcessingConfig, ResourceManager, StreamlineFileHandler, HFConversionEngine
        from llm.gemini_connector import GeminiConnectionManager  
        from visualization.analytics_engine import QualityMetricsCalculator
        
        config = ProcessingConfig()
        resource_manager = ResourceManager(config)
        file_handler = StreamlineFileHandler(resource_manager)
        conversion_engine = HFConversionEngine(resource_manager, config)
        gemini_manager = GeminiConnectionManager()
        quality_calculator = QualityMetricsCalculator()
        
        orchestrator = DocumentProcessingOrchestrator(
            file_handler, conversion_engine, gemini_manager, quality_calculator
        )
        
        service_layer = create_hf_spaces_service_layer(orchestrator)
        health = service_layer.get_system_health()
        print(f"   ✅ Service Layer: {health['overall_status']}")
        
        # Тест 2: UI Components
        print("2️⃣ Тестування UI Components...")
        ui_factory = UIComponentFactory({'title': 'Test Platform'})
        processing_tab = ui_factory.create_processing_tab()
        print("   ✅ UI Components готові")
        
        # Тест 3: Response Factory
        print("3️⃣ Тестування Response Factory...")
        response_factory = StrategicResponseFactory()
        metrics = response_factory.get_factory_metrics()
        print(f"   ✅ Response Factory: {metrics['factory_health']}")
        
        # Тест 4: Event System
        print("4️⃣ Тестування Event System...")
        event_orchestrator = EventOrchestrator(service_layer, response_factory)
        event_metrics = event_orchestrator.get_orchestrator_metrics()
        print(f"   ✅ Event System: {len(event_metrics['handlers_registered'])} handlers")
        
        print("\n🎉 Всі компоненти архітектури працюють правильно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Помилка в архітектурі: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_architecture())
    exit(0 if success else 1)