import os
import torch
from src.config import Constants
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Optional

class LLMModel:
    """
    Optimized language model implementation with memory-efficient features.
    """
    def __init__(
        self,
        model_name: str,
        hf_token: str = Constants.HF_TOKEN,
        device: str = Constants.DEVICE,
        load_in_8bit: bool = Constants.LOAD_IN_8BIT,
        use_half_precision: bool = Constants.USE_HALF_PRECISION,
        max_memory: Optional[Dict[str, str]] = Constants.MAX_MEMORY
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.use_half_precision = use_half_precision
        self.max_memory = max_memory
        
        self.local_llm_path = os.path.join(Constants.LLM_MODEL_PATH, model_name.replace("/", "_"))
        self.local_tkn_path = os.path.join(Constants.TOKENIZER_PATH, model_name.replace("/", "_"))

        # Load components with memory optimizations
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.pipeline = self._load_pipeline()
        
    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Загружает токенизатор из локального пути, если доступен.
        Иначе скачивает и сохраняет на диск.
        """

        if os.path.exists(self.local_tkn_path):
            print(f"Loading tokenizer from local path: {self.local_tkn_path}")
            return AutoTokenizer.from_pretrained(
                self.local_tkn_path,
                use_fast=True,  # Use fast tokenizer
                # model_max_length=512  # Limit max length
            )
        else:
            print(f"Downloading tokenizer: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                # model_max_length=512
            )
            os.makedirs(self.local_tkn_path, exist_ok=True)
            tokenizer.save_pretrained(self.local_tkn_path)
            return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """
        Загружает модель из локального пути, если доступна.
        Иначе скачивает с Hugging Face, сохраняет и переносит на нужное устройство.
        """
        load_kwargs = {
            "device_map": self.device,
            "max_memory": self.max_memory,
            "token": self.hf_token
        }
        
        # Add quantization if enabled
        if self.load_in_8bit:
            load_kwargs.update({
                "load_in_8bit": True,
                "torch_dtype": torch.float16
            })
        elif self.use_half_precision:
            load_kwargs["torch_dtype"] = torch.float16
            
        if os.path.exists(self.local_llm_path):
            print(f"Loading model from local path: {self.local_llm_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.local_llm_path,
                **load_kwargs
            )
        else:
            print(f"Downloading model: {self.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            model.save_pretrained(self.local_llm_path)
            
        # Enable memory efficient features
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        return model

    def _load_pipeline(self):
        """Load optimized pipeline."""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

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
        # Use inference mode and automatic mixed precision
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.use_half_precision):
            # Convert text to tensors efficiently
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                truncation=True,
                # max_length=512,  # Limit input length
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate with optimized settings
            output_ids = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV-cache
            )

            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode and return
            return self.tokenizer.decode(output_ids[0][inputs.size(1):], skip_special_tokens=True)

    def __del__(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
