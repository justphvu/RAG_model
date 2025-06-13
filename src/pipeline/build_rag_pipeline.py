from src.config import Constants
from src.models.embedding import EmbeddingModel
from src.models.llm import LLMModel
from src.pipeline.ragpipeline import RAGPipeline
from src.preprocess.db_manager import ChromaDBManager
from src.retrieval.retriever import Retriever

def build_rag_pipeline() -> RAGPipeline:
    # Step 1: Build chunker, embedding model, LM model, etc.
    embedding_model = EmbeddingModel(Constants.EMBEDDING_MODEL)
    llm = LLMModel(Constants.LLM_MODEL)

    chroma_manager = ChromaDBManager(
        persist_directory=Constants.DB_PATH,
        embedding_model=embedding_model
    )
    retriever = Retriever(
        embedding_model=embedding_model,
        chroma_manager=chroma_manager,
        pipeline=llm.pipeline
    )

    return RAGPipeline(
        retriever=retriever,
        llm=llm
    )

