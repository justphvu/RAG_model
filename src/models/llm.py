import os
import torch
import logging
from typing import Optional, Dict, Any, List
from src.config import Constants
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class LLMModel:
    """
    A wrapper for a Hugging Face causal language model and tokenizer.

    This class handles loading the model and tokenizer from local paths or
    downloading them from the Hugging Face Hub. It provides a simple
    interface for text generation and includes error handling.
    """
    def __init__(
        self,
        model_name: str,
        hf_token: str = Constants.HF_TOKEN,
        device: str = Constants.DEVICE
    ) -> None:
        """
        Initializes the LLMModel.

        Loads the model, tokenizer, and sets up the generation pipeline.
        Errors during loading are logged, and the respective components
        will be set to None.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub
                or a local path.
            hf_token (str): The Hugging Face API token for private models.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.local_llm_path = os.path.join(Constants.LLM_MODEL_PATH, model_name.replace("/", "_"))
        self.local_tkn_path = os.path.join(Constants.TOKENIZER_PATH, model_name.replace("/", "_"))

        try:
            self.model: Optional[AutoModelForCausalLM] = self._load_model()
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
        try:
            self.tokenizer: Optional[AutoTokenizer] = self._load_tokenizer()
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
        try:
            self.pipeline: Optional[pipeline] = self._load_pipeline()
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.pipeline = None

    def _load_tokenizer(self) -> Optional[AutoTokenizer]:
        """
        Loads the tokenizer, downloading if not found locally.

        Returns:
            Optional[AutoTokenizer]: The loaded tokenizer instance, or None if an error occurred.
        """
        try:
            if os.path.exists(self.local_tkn_path):
                logger.info(f"Loading tokenizer from local path: {self.local_tkn_path}")
                return AutoTokenizer.from_pretrained(self.local_tkn_path, use_fast=False)
            else:
                logger.info(f"Downloading tokenizer: {self.model_name}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
                os.makedirs(self.local_tkn_path, exist_ok=True)
                tokenizer.save_pretrained(self.local_tkn_path)
                logger.info(f"Tokenizer saved to: {self.local_tkn_path}")
                return tokenizer
        except Exception as e:
            logger.error(f"Tokenizer loading failed: {e}", exc_info=True)
            return None

    def _load_model(self) -> Optional[AutoModelForCausalLM]:
        """
        Loads the model, downloading if not found locally.

        Returns:
            Optional[AutoModelForCausalLM]: The loaded model instance, or None if an error occurred.
        """
        try:
            if os.path.exists(self.local_llm_path):
                logger.info(f"Loading model from local path: {self.local_llm_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.local_llm_path,
                    device_map=self.device,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16
                )
            else:
                logger.info(f"Downloading model: {self.model_name}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16
                )
                model.save_pretrained(self.local_llm_path)
                logger.info(f"Model saved to: {self.local_llm_path}")
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            return None

    def _load_pipeline(self) -> Optional[pipeline]:
        """
        Creates a text-generation pipeline with the loaded model and tokenizer.

        Returns:
            Optional[pipeline]: The text generation pipeline instance, or None if model/tokenizer is missing.
        """
        try:
            if self.model is not None and self.tokenizer is not None:
                return pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            else:
                logger.error("Cannot create pipeline: model or tokenizer is not loaded.")
                return None
        except Exception as e:
            logger.error(f"Pipeline loading failed: {e}", exc_info=True)
            return None

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = Constants.max_new_tokens,
        temperature: float = Constants.temperature,
        top_k: int = Constants.top_k,
        top_p: float = Constants.top_p
    ) -> str:
        """
        Generates text based on an input prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The input prompt. Can be a single
                string or a list of dictionaries for chat-based models.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The value used to module the next token probabilities.
            top_k (int): The number of highest probability vocabulary tokens to keep for
                top-k-filtering.
            top_p (float): If set to float < 1, only the most probable tokens with
                probabilities that add up to top_p or higher are kept for generation.

        Returns:
            str: The generated text, or an error message string if generation fails.
        """
        if self.model is None or self.tokenizer is None or self.pipeline is None:
            logger.error("Cannot generate text because model, tokenizer, or pipeline is not loaded.")
            return "[LLM not initialized]"
        try:
            # The pipeline can handle chat templates directly.
            result = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # Extract the generated text from the pipeline's output
            if isinstance(result, list) and result and 'generated_text' in result[0]:
                if isinstance(prompt, list): # chat format
                     # find where the response starts
                    full_text = result[0]['generated_text']
                    # This is a bit brittle, but works for many models
                    response_start = full_text.rfind("[/INST]") + len("[/INST]") if "[/INST]" in full_text else -1
                    return full_text[response_start:].strip()
                else:
                    return result[0]['generated_text']
            return str(result)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return "[LLM generation error]"
