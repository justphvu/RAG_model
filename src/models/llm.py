import os
import torch
from src.config import Constants
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMModel:
    """
    Инициализирует языковую модель, токенизатор и их пайлайн.
    Если модель доступна локально — загружается из ./models/lm/{model_name},
    иначе скачивается и сохраняется в эту папку.

    Args
    model_name (str): название модели из Hugging Face или локальный идентификатор
    token (str): токон из Hugging Face
    device (str): 'cuda', 'cpu' или None (автовыбор)
    """
    def __init__(
        self,
        model_name: str,
        hf_token: str = Constants.HF_TOKEN,
        device: str = Constants.DEVICE
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.local_llm_path = os.path.join(Constants.LLM_MODEL_PATH, model_name.replace("/", "_"))
        self.local_tkn_path = os.path.join(Constants.TOKENIZER_PATH, model_name.replace("/", "_"))

        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.pipeline = self._load_pipeline()
        
    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Загружает токенизатор из локального пути, если доступен.
        Иначе скачивает и сохраняет на диск.
        """
        if os.path.exists(self.local_tkn_path):
            print(f"Loading tokenizer from local path: {self.local_tkn_path}")
            return AutoTokenizer.from_pretrained(self.local_tkn_path, use_fast=False)
        else:
            print(f"Downloading tokenizer: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            os.makedirs(self.local_tkn_path, exist_ok=True)
            tokenizer.save_pretrained(self.local_tkn_path)
            print(f"Tokenizer saved to: {self.local_tkn_path}")
            return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """
        Загружает модель из локального пути, если доступна.
        Иначе скачивает с Hugging Face, сохраняет и переносит на нужное устройство.
        """
        if os.path.exists(self.local_llm_path):
            print(f"Loading model from local path: {self.local_llm_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.local_llm_path,
                device_map=self.device,
                token=self.hf_token,
                torch_dtype=torch.bfloat16
            )
        else:
            print(f"Downloading model: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                token=self.hf_token,
                torch_dtype=torch.bfloat16
            )
            model.save_pretrained(self.local_llm_path)
            print(f"Model saved to: {self.local_llm_path}")
        return model.to(self.device)

    def _load_pipeline(self):
        return pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = Constants.max_new_tokens,
        temperature: float = Constants.temperature,
        top_k: int = Constants.top_k,
        top_p: float = Constants.top_p
    ) -> str:
        """
        Генерирует текст на основе входного запроса.

        Args
        prompt (str): входной текст-запрос
        max_new_tokens (int): макс. кол-во новых токенов, которые можно сгенерировать
        temperature (float): уровень случайности (0.0 — детерминированно, >1.0 — более разнообразно)
        top_k (int): уровень ограниченности токоны-кандидаты (0 — детерминированно, >50 — более разнообразно)
        top_p (float): динамический выбор токена (0.0 — детерминированно, >1.0 — более разнообразно)
        Returns:
            str: сгенерированный текст
        """
        # Преобразуем текст в тензоры
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            truncation=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # Генерируем выход
        output_ids = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id  # чтобы избежать предупреждения
        )

        # Декодируем тензоры в строку
        return self.tokenizer.decode(output_ids[0][inputs.size(1) :], skip_special_tokens=True)
