#!/usr/bin/env python3
"""
Test script to demonstrate and validate the performance optimizations.

This script tests:
1. Enhanced embedding model with caching and batching
2. Conversation history management
3. Enhanced RAG pipeline with conversation support
4. Performance monitoring
"""

import time
import asyncio
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_optimizations():
    """Test the enhanced embedding model optimizations."""
    logger.info("🧪 Testing Embedding Model Optimizations...")
    
    from src.models.embedding import OptimizedEmbeddingModel
    
    # Initialize optimized embedding model
    embedding_model = OptimizedEmbeddingModel(
        model_name="ai-forever/FRIDA",
        device="cpu",  # Use CPU for testing
        cache_dir="./cache/embeddings",
        batch_size=16,
        enable_disk_cache=True
    )
    
    # Test single embedding with caching
    logger.info("Testing single embedding with caching...")
    start_time = time.time()
    
    # First call (should be slow)
    embedding1 = embedding_model.embed("What are the admission requirements?", "search_query")
    first_call_time = time.time() - start_time
    
    # Second call (should be fast due to caching)
    start_time = time.time()
    embedding2 = embedding_model.embed("What are the admission requirements?", "search_query")
    second_call_time = time.time() - start_time
    
    logger.info(f"First call time: {first_call_time:.3f}s")
    logger.info(f"Second call time: {second_call_time:.3f}s")
    logger.info(f"Speedup: {first_call_time/second_call_time:.1f}x")
    
    # Test batch processing
    logger.info("Testing batch processing...")
    texts = [
        "What are the admission requirements?",
        "How much does the program cost?",
        "What are the application deadlines?",
        "Tell me about the curriculum",
        "What are the job prospects?",
        "How long is the program?",
        "What are the prerequisites?",
        "Tell me about the faculty"
    ]
    
    start_time = time.time()
    batch_embeddings = embedding_model.embed_batch(texts, "search_query")
    batch_time = time.time() - start_time
    
    logger.info(f"Batch processing time: {batch_time:.3f}s")
    logger.info(f"Average time per text: {batch_time/len(texts):.3f}s")
    
    # Get cache statistics
    cache_stats = embedding_model.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")
    
    return embedding_model

def test_conversation_management():
    """Test the conversation history management."""
    logger.info("🧪 Testing Conversation Management...")
    
    from src.models.conversation import ConversationManager
    
    # Initialize conversation manager
    conv_manager = ConversationManager(
        max_conversations=100,
        conversation_ttl=3600,
        enable_persistence=True,
        persistence_file="./cache/test_conversations.json"
    )
    
    # Test conversation flow
    user_id = "test_user_123"
    
    # Add some messages
    conv_manager.add_user_message(user_id, "What are the admission requirements?")
    conv_manager.add_assistant_message(user_id, "The admission requirements include...")
    conv_manager.add_user_message(user_id, "How much does it cost?")
    conv_manager.add_assistant_message(user_id, "The program costs...")
    conv_manager.add_user_message(user_id, "What about scholarships?")
    conv_manager.add_assistant_message(user_id, "There are several scholarship options...")
    
    # Get conversation history
    history = conv_manager.get_conversation_history(user_id, max_messages=5)
    logger.info(f"Conversation history length: {len(history)}")
    
    # Get context window
    context = conv_manager.get_context_window(user_id, max_tokens=1000)
    logger.info(f"Context window length: {len(context)}")
    
    # Get statistics
    stats = conv_manager.get_stats()
    logger.info(f"Conversation stats: {stats}")
    
    return conv_manager

def test_enhanced_rag_pipeline():
    """Test the enhanced RAG pipeline with conversation support."""
    logger.info("🧪 Testing Enhanced RAG Pipeline...")
    
    from src.pipeline.build_rag_pipeline import build_rag_pipeline
    
    # Build enhanced pipeline
    rag_pipeline = build_rag_pipeline()
    
    # Test queries
    test_queries = [
        "What are the admission requirements for Computer Science?",
        "How much does the Master's program cost?",
        "Tell me about PhD programs",
        "What are the application deadlines?",
        "Are there any scholarships available?"
    ]
    
    user_id = "test_user_456"
    
    for i, query in enumerate(test_queries):
        logger.info(f"Processing query {i+1}: {query}")
        start_time = time.time()
        
        try:
            response = rag_pipeline.generate_answer(
                query=query,
                user_id=user_id,
                use_conversation_history=True
            )
            processing_time = time.time() - start_time
            
            logger.info(f"Response time: {processing_time:.3f}s")
            logger.info(f"Response length: {len(response)} characters")
            logger.info(f"Response preview: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    # Get performance statistics
    cache_stats = rag_pipeline.get_cache_stats()
    perf_stats = rag_pipeline.get_performance_stats()
    conv_stats = rag_pipeline.get_conversation_stats()
    
    logger.info(f"Cache stats: {cache_stats}")
    logger.info(f"Performance stats: {perf_stats}")
    logger.info(f"Conversation stats: {conv_stats}")
    
    return rag_pipeline

def test_performance_monitoring():
    """Test the performance monitoring system."""
    logger.info("🧪 Testing Performance Monitoring...")
    
    from src.utils.performance_monitor import PerformanceMonitor
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(
        max_history_size=100,
        monitoring_interval=2.0,
        enable_monitoring=True
    )
    
    # Add some custom metrics
    monitor.add_custom_metric("embedding_cache_hit_rate", 0.85)
    monitor.add_custom_metric("avg_response_time", 2.3)
    monitor.add_custom_metric("active_users", 15)
    
    # Wait for some metrics to be collected
    time.sleep(5)
    
    # Get current metrics
    current_metrics = monitor.get_current_metrics()
    logger.info(f"Current CPU usage: {current_metrics.cpu_percent:.1f}%")
    logger.info(f"Current memory usage: {current_metrics.memory_percent:.1f}%")
    logger.info(f"Active threads: {current_metrics.active_threads}")
    
    # Get average metrics
    avg_metrics = monitor.get_average_metrics(duration_minutes=1)
    logger.info(f"Average metrics (1 min): {avg_metrics}")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    logger.info(f"Performance summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    return monitor

async def test_bot_integration():
    """Test the enhanced bot integration."""
    logger.info("🧪 Testing Bot Integration...")
    
    from src.interface.bot import build_bot_handler
    from src.pipeline.build_rag_pipeline import build_rag_pipeline
    
    # Build enhanced pipeline
    rag_pipeline = build_rag_pipeline()
    
    # Test bot handler function
    bot_handler = build_bot_handler(rag_pipeline)
    
    # Simulate a message update (this is a simplified test)
    class MockUpdate:
        def __init__(self, text, user_id):
            self.message = MockMessage(text, user_id)
            self.effective_user = MockUser(user_id)
            self.effective_chat = MockChat()
    
    class MockMessage:
        def __init__(self, text, user_id):
            self.text = text
            self.chat = MockChat()
    
    class MockUser:
        def __init__(self, user_id):
            self.id = user_id
    
    class MockChat:
        def __init__(self):
            self.id = "test_chat"
    
    class MockContext:
        def __init__(self):
            self.bot = MockBot()
        
        async def send_chat_action(self, chat_id, action):
            pass
    
    class MockBot:
        async def send_chat_action(self, chat_id, action):
            pass
    
    # Test with a mock update
    mock_update = MockUpdate("What are the admission requirements?", "test_user_789")
    mock_context = MockContext()
    
    logger.info("Testing bot handler with mock update...")
    start_time = time.time()
    
    try:
        await bot_handler(mock_update, mock_context)
        processing_time = time.time() - start_time
        logger.info(f"Bot handler processing time: {processing_time:.3f}s")
    except Exception as e:
        logger.error(f"Error in bot handler: {e}")

def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting Performance Optimization Tests...")
    
    try:
        # Test embedding optimizations
        embedding_model = test_embedding_optimizations()
        
        # Test conversation management
        conv_manager = test_conversation_management()
        
        # Test enhanced RAG pipeline
        rag_pipeline = test_enhanced_rag_pipeline()
        
        # Test performance monitoring
        monitor = test_performance_monitoring()
        
        # Test bot integration
        asyncio.run(test_bot_integration())
        
        logger.info("✅ All optimization tests completed successfully!")
        
        # Print summary
        logger.info("\n📊 Optimization Summary:")
        logger.info("• Enhanced embedding model with caching and batching")
        logger.info("• Conversation history management with persistence")
        logger.info("• Enhanced RAG pipeline with conversation support")
        logger.info("• Performance monitoring and metrics collection")
        logger.info("• Improved bot interface with new commands")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 