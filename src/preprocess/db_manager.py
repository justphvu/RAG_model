import os
from typing import List, Dict, Optional
from chromadb import PersistentClient

class ChromaDBManager:
    """
    Инициализирует ChromaDB PersistentClient, создает (или получает) две коллекции:
    одну для родительских чанков, вторую — для дочерних.

    Args
    persist_directory: путь для хранения базы ChromaDB
    embedding_model: модель, которая умеет генерировать эмбеддинги
    parent_collection_name: название коллекции родительских чанков
    child_collection_name: название коллекции дочерних чанков
    """
    def __init__(
        self,
        persist_directory: str,
        embedding_model,
        parent_collection_name: str = "parent_docs",
        child_collection_name: str = "child_docs"
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = PersistentClient(path=persist_directory)
        self.embedding_model = embedding_model

        self.parent_collection = self.client.get_or_create_collection(name=parent_collection_name)
        self.child_collection = self.client.get_or_create_collection(name=child_collection_name)

    def upsert_documents(self, docs: List[Dict], collection_type: str = "child"):
        """
        Добавляет или обновляет документы в указанной коллекции (родительской или дочерней).

        Args
        docs: список словарей, содержащих поля "id", "text", "metadata"
        collection_type: тип коллекции ("child" или "parent")
        """
        assert collection_type in {"child", "parent"}, "Invalid collection_type"

        collection = self.child_collection if collection_type == "child" else self.parent_collection

        for doc in docs:
            text = doc["text"]
            embedding = self.embedding_model.embed(text, prompt_name="search_document")
            collection.upsert(
                ids=[doc["id"]],
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc["metadata"]]
            )

    def persist(self):
        """
        Метод-заглушка: при использовании PersistentClient данные сохраняются автоматически.
        Можно расширить для логики сброса, архивирования и т.п.
        """
        pass

    def reset_collections(self):
        """
        Удаляет и пересоздает обе коллекции — полезно для повторной загрузки данных.
        """
        self.client.delete_collection(name=self.parent_collection.name)
        self.client.delete_collection(name=self.child_collection.name)
        self.parent_collection = self.client.get_or_create_collection(name=self.parent_collection.name)
        self.child_collection = self.client.get_or_create_collection(name=self.child_collection.name)
