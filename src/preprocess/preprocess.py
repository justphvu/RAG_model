from src.config import Constants
from src.models.embedding import EmbeddingModel
from src.preprocess.chunking import ChunkBuilder
from src.preprocess.html_parser import HTMLPreprocess
from src.preprocess.db_manager import ChromaDBManager

def preprocess() -> None:
    """
    Главная функция предобработки:
    - Парсит HTML-документы
    - Разбивает их на родительские и дочерние чанки
    - Генерирует эмбеддинги
    - Сохраняет данные в ChromaDB
    """
    # Шаг 1: Загрузка и парсинг HTML-документов
    html_processor = HTMLPreprocess(Constants.RAW_HTML_PATH)
    processed_documents = html_processor.preprocess()

    # Шаг 2: Создание построителя чанков
    chunk_builder = ChunkBuilder()
    
    # Шаг 3: Загрузка модели эмбеддингов
    embedding_model = EmbeddingModel(Constants.EMBEDDING_MODEL)

    # Шаг 4: Инициализация менеджера ChromaDB с двумя коллекциями
    chroma_manager = ChromaDBManager(
        persist_directory=Constants.DB_PATH,
        embedding_model=embedding_model
    )

    # Шаг 5: Обработка каждого файла и вставка в БД
    for i, file_data in enumerate(processed_documents):
        print(f"Upserting file_{i+1}")
        parent_docs, child_docs = chunk_builder.build_chunks_with_context(file_data)
        chroma_manager.upsert_documents(child_docs, collection_type=Constants.CHILD_COLLECTION)
        chroma_manager.upsert_documents(parent_docs, collection_type=Constants.PARENT_COLLECTION)

if __name__=="__main__":
    preprocess()
