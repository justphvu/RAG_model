from src.config import Constants
from src.models.embedding import OptimizedEmbeddingModel
from src.models.llm import LLMModel
from src.pipeline.ragpipeline import EnhancedRAGPipeline
from src.preprocess.db_manager import ChromaDBManager
from src.retrieval.retriever import Retriever
from src.models.conversation import ConversationManager

def build_rag_pipeline() -> EnhancedRAGPipeline:
    """
    Build an enhanced RAG pipeline with conversation support and performance optimizations.
    
    Returns:
        EnhancedRAGPipeline: Configured RAG pipeline with all optimizations
    """
    # Step 1: Build optimized embedding model with caching and batching
    embedding_model = OptimizedEmbeddingModel(
        model_name=Constants.EMBEDDING_MODEL,
        device=Constants.DEVICE,
        cache_dir=Constants.EMBEDDING_CACHE_DIR,
        batch_size=Constants.BATCH_SIZE,
        enable_disk_cache=Constants.ENABLE_DISK_CACHE,
    )
    
    # Step 2: Build LLM model
    llm = LLMModel(Constants.LLM_MODEL)

    # Step 3: Initialize ChromaDB manager with optimized embedding model

    chroma_manager = ChromaDBManager(
        persist_directory=Constants.DB_PATH,
        embedding_model=embedding_model
    )
    
    # Step 4: Build retriever with optimized components
    retriever = Retriever(
        embedding_model=embedding_model,
        chroma_manager=chroma_manager,
        pipeline=llm.pipeline
    )

    # Step 5: Initialize conversation manager
    conversation_manager = ConversationManager(
        max_conversations=Constants.MAX_CONVERSATIONS,
        conversation_ttl=Constants.CONVERSATION_TTL,
        enable_persistence=Constants.ENABLE_PERSISTENCE,
        persistence_file=Constants.PERSISTENCE_FILE
    )

    # Step 6: Build enhanced RAG pipeline
    return EnhancedRAGPipeline(
        retriever=retriever,
        llm=llm,
        conversation_manager=conversation_manager,
        device=Constants.DEVICE,
        enable_query_cache=True,
        max_context_tokens=Constants.MAX_CONTEXT_TOKENS
    )

