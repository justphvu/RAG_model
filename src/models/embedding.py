import os
from src.config import Constants
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    Обертка вокруг SentenceTransformer для встраивания списка текстов.
    Загружает модель из локального пути, если он доступен, в противном случае из Hugging Face.

    Args:
    model_name (str): Имя модели SentenceTransformer.
    device (str): Устройство для запуска вывода, например, 'cuda:0' или 'cpu'.
    """
    def __init__(self, model_name: str, device: str = Constants.DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        local_path = os.path.join(Constants.EMBEDDING_MODEL_PATH, self.model_name)
        if os.path.exists(local_path):
            print(f"Loading model from local path: {local_path}")
            return SentenceTransformer(local_path, device=self.device)
        else:
            print(f"Loading model from Hugging Face: {self.model_name}")
            embedding_model = SentenceTransformer(self.model_name, device=self.device)
            os.makedirs(local_path, exist_ok=True)
            embedding_model.save(local_path)
            return embedding_model

    def embed(self, texts: List, prompt_name: str) -> List[List]:
        """
        Энкодит список текстов на эмбеддинги

        Args:
            texts (List[str]): Список входных текстов.
            prompt_name (str): Тип эмбеддинга

        Returns:
            List[List[float]]: Список векторов эмбеддингов.
        """
        if not texts:
            return []
        return self.model.encode(texts, prompt_name=prompt_name)
