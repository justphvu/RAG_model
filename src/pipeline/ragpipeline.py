import json
from typing import List
from src.config import Constants

class RAGPipeline:
    """
    Инициализирует RAGPipeline для ответа на запросы
    
    Args:
    retriever (Retriever): объект метода RAG
    llm (LLMModel): объект LLM
    device (str): Устройство для запуска вывода, например, 'cuda:0' или 'cpu'.
    """
    def __init__(
        self,
        retriever,
        llm,
        device: str = Constants.DEVICE
    ):
        self.retriever = retriever
        self.llm = llm
        self.device = device

    def generate_prompt(
        self,
        question: str,
        answers: List[dict],
        system_prompt: str = Constants.DEFAULT_SYSTEM_PROMPT
    ) -> str:
        """
        Формирует разговор (чат) из списка документов-ответов и вопроса.

        Args:
            answers (List[Dict]): Список отранжированных документов.
            question (str): Вопрос пользователя.
            system_prompt (str): Системный промпт для модели.

        Returns:
            List[Dict]: Список сообщений в формате чата: system + user.
        """
        formatted_docs = []
        for i, answer in enumerate(answers):
            metadata = answer.get("metadata", {})
            doc_type = metadata.get("type")
            title = metadata.get("title", "No Title")

            if doc_type == "table":
                text = metadata.get("html", "")
            elif doc_type == "text":
                text = answer.get("text", "")
            else:
                continue  # Пропустить неизвестный тип

            if not text.strip():
                continue  # Пропустить пустые документы

            formatted_docs.append({
                "doc_id": i,
                "title": title,
                "content": text
            })

        query = f"""
        {json.dumps(formatted_docs, ensure_ascii=False)}
        На основе выше документов ответь на следующий вопрос:
        {question}
        """
        # Формируем системный промпт
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        return prompt

    def generate_answer(self, query: str, top_k: int = 10) -> str:
        """
        Отвечает на пользовательский запрос с использованием RAG-пайплайна.

        Шаги:
        1. Генерация альтернативных формулировок запроса.
        2. Поиск дочерних документов (child chunks) в векторной базе.
        3. Получение родительских документов по ID.
        4. Реранжировка документов по косинусной близости.
        5. Формирование чата и генерация ответа моделью.

        Returns:
            answer (str): Сгенерированный ответ модели.
            reranked_docs (List[Dict]): Документы, использованные для ответа.
        """
        # Step 1: Generate alternative queries
        alt_queries = self.retriever.generate_alternative_queries(query)

        # Step 2: Retrieve relevant child docs
        child_docs = self.retriever.query_vectorstore(alt_queries)

        # Step 3: Get parent docs
        parent_docs = self.retriever.retrieve_parent_docs(child_docs)

        # Step 4: Rerank
        reranked_docs = self.retriever.rerank(query, parent_docs, top_k=top_k)

        # Step 5: Build prompt
        prompt = self.generate_prompt(query, reranked_docs)

        # Step 6: Generate answer
        answer = self.llm.generate(prompt)
        # input_ids = self.tokenizer.apply_chat_template(prompt, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # outputs = self.model.generate(
        #         input_ids=input_ids,
        #         max_new_tokens=768,
        #         do_sample=True,
        #         temperature=0.7,
        #         top_k=50,
        #         top_p=0.95,
        #         pad_token_id=self.tokenizer.eos_token_id
        # )
        # answer = self.tokenizer.decode(outputs[0][input_ids.size(1) :], skip_special_tokens=True)
        return answer
