from typing import List, Dict

import numpy as np

class Retriever:
    """Общий флоу:
    1. Пользователь задаёт запрос.
    2. LLM генерирует альтернативу.
    3. Запросы кодируются и ищутся в child_collection.
    4. Извлекаются parent_ids и подгружаются родительские документы.
    5. Родительские документы реранжируются по близости.
    
    Args:
    embedding_model: модель эмбеддинга
    chroma_manager: клиент векторной базы данных
    pipeline: RAG-пайлайн
    """
    def __init__(
        self,
        embedding_model,
        chroma_manager,
        pipeline
    ):
        self.embedding_model = embedding_model
        self.chroma_manager = chroma_manager
        self.pipe = pipeline

    # Generate alternative queries (basic example, since langchain’s MultiQueryRetriever uses LLMs)
    def generate_alternative_queries(self, original_query: str) -> List[str]:
        """
        Использует pipe (LLM), чтобы сгенерировать альтернативные формулировки запроса.
        Возвращает список: [оригинал, альтернатива_1, ...]

        Args:
            original_query (str): Исходный запрос.
        """
        prompt = [
            {"role": "user", "content" : f"Сформулируй другой вариант вопроса:\nВопрос: {original_query}"}
        ]
        result = self.pipe(prompt, do_sample=False, temperature=0.9, max_length=64)[0]["generated_text"]
        return [original_query, result[1]['content']]

    # Embed queries and search in Chroma
    def query_vectorstore(
        self,
        queries: List[str],
        per_query_k=100,
        final_k=200,
        score_threshold=1.5
    ) -> List[Dict]:
        """
        Выполняет поиск по ChromaDB для каждого запроса.
        Возвращает отсортированные результаты, прошедшие порог расстояния.
        Если ничего не найдено, повторяет запрос без фильтра по расстоянию.

        Args:
            queries (List[str]): Список строк-запросов.
            child_collection: Коллекция ChromaDB.
            embedding_model: Модель эмбеддингов с методом .encode().
            per_query_k (int): Сколько документов запрашивать на каждый запрос.
            final_k (int): Сколько документов вернуть в итоге.
            score_threshold (float): Максимально допустимое расстояние (чем меньше, тем релевантнее).
        """
        all_results = []
        for query in queries:
            query_embedding = self.embedding_model.embed(query, prompt_name="search_query")
            results = self.chroma_manager.child_collection.query(
                query_embeddings=[query_embedding],
                n_results=per_query_k
            )
            # Transform result into dicts
            for doc, score, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                if score <= score_threshold:
                    all_results.append({
                        "text": doc,
                        "score": score,
                        "metadata": meta
                    })

        # Если ничего не найдено по порогу — повторяем без фильтрации
        if not all_results:
            print("Ни один результат не прошёл по порогу. Возвращаем менее релевантные документы.")
            for query in queries:
                query_embedding = self.embedding_model.embed(query, prompt_name="search_query")
                results =  self.chroma_manager.child_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=per_query_k
                )

                for doc, score, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                    all_results.append({
                        "text": doc,
                        "score": score,
                        "metadata": meta
                    })

        sorted_docs = sorted(all_results, key=lambda x: x["score"])
        return sorted_docs[:final_k]

    # Map child to parent
    def retrieve_parent_docs(self, child_results: List[Dict]) -> List[Dict]:
        """
        Группирует дочерние документы по 'parent_id' и извлекает соответствующие родительские документы.

        Args:
            child_results: Список результатов дочерней коллекции (каждый содержит metadata с parent_id).
            parent_collection: Коллекция ChromaDB, содержащая родительские документы.

        Returns:
            Список словарей с родительскими документами (id, text, embedding, metadata).
        """
        parent_ids = list({
            res["metadata"]["parent_id"]
            for res in child_results
            if res.get("metadata") and "parent_id" in res["metadata"]
        })

        if not parent_ids:
            return []  # Нечего извлекать

        parent_collection = self.chroma_manager.parent_collection

        retrieved = parent_collection.get(
            ids=parent_ids,
            include=["documents", "embeddings", "metadatas"]
        )

        parent_chunks = []
        for i, parent_id in enumerate(retrieved["ids"]):
            parent_chunks.append({
                "id": parent_id,
                "text": retrieved["documents"][i],
                "embedding": retrieved["embeddings"][i],
                "metadata": retrieved["metadatas"][i]
            })
        return parent_chunks

    # Rerank parent chunks (rudimentary similarity reranking)
    def rerank(
        self,
        query: str,
        parent_chunks: List[Dict],
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Реранжирует родительские чанки по косинусной близости к запросу.

        Args:
            query: Строка-запрос.
            parent_chunks: Список документов с полем "embedding".
            top_k: Максимальное количество возвращаемых результатов.
            threshold: Минимально допустимая близость (0–1).

        Returns:
            Список чанков, отсортированных по убыванию близости.
        """
        if not parent_chunks:
            return []

        embeddings = np.array([doc["embedding"] for doc in parent_chunks])
        query_vec = np.array(self.embedding_model.embed(query, prompt_name="search_query"))

        # Вектор косинусной близости
        dot_products = embeddings @ query_vec
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        similarities = dot_products / norms

        # Фильтрация по threshold
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) > 0:
            # Фильтрация по threshold
            filtered = [{
                "text": parent_chunks[i]["text"],
                "score": float(similarities[i]),
                "metadata": parent_chunks[i]["metadata"],
            } for i in valid_indices]

            # Сортируем по убыванию score и возвращаем top_k
            sorted_filtered = sorted(filtered, key=lambda x: -x["score"])
            return sorted_filtered[:top_k]
        else:
            # Вернуть top_k самых похожих документов, несмотря на низкий score
            top_indices = np.argsort(-similarities)[:top_k]
            fallback = [{
                "text": parent_chunks[i]["text"],
                "score": float(similarities[i]),
                "metadata": parent_chunks[i]["metadata"],
                "id": parent_chunks[i].get("id")
            } for i in top_indices]
            return fallback
