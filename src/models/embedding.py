import os
import hashlib
import pickle
from functools import lru_cache
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import Constants

class OptimizedEmbeddingModel:
    """
    Enhanced embedding model with caching, batching, and performance optimizations.
    
    Features:
    - LRU caching for frequently used embeddings
    - Batch processing for efficiency
    - Disk-based caching for persistent storage
    - Memory-efficient processing
    """
    
    def __init__(
        self, 
        model_name: str, 
        device: str = Constants.DEVICE,
        cache_dir: str = "./cache/embeddings",
        batch_size: int = 32,
        enable_disk_cache: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.enable_disk_cache = enable_disk_cache
        
        # Create cache directory
        if self.enable_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize in-memory cache
        self._memory_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model from local path or Hugging Face."""
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
    
    def _generate_cache_key(self, text: str, prompt_name: str) -> str:
        """Generate a unique cache key for text and prompt combination."""
        content = f"{text}:{prompt_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> str:
        """Get the disk cache file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from disk cache if available."""
        if not self.enable_disk_cache:
            return None
        
        cache_path = self._get_disk_cache_path(cache_key)
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading from disk cache: {e}")
        return None
    
    def _save_to_disk_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to disk cache."""
        if not self.enable_disk_cache:
            return
        
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Error saving to disk cache: {e}")
    
    @lru_cache(maxsize=1000)
    def embed_single_cached(self, text: str, prompt_name: str) -> List[float]:
        """
        Embed a single text with caching (in-memory LRU cache).
        
        Args:
            text: Input text to embed
            prompt_name: Type of embedding prompt
            
        Returns:
            Embedding vector
        """
        cache_key = self._generate_cache_key(text, prompt_name)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            self._cache_stats["hits"] += 1
            return self._memory_cache[cache_key]
        
        # Check disk cache
        disk_embedding = self._load_from_disk_cache(cache_key)
        if disk_embedding is not None:
            self._memory_cache[cache_key] = disk_embedding
            self._cache_stats["hits"] += 1
            return disk_embedding
        
        # Generate new embedding
        self._cache_stats["misses"] += 1
        embedding = self.model.encode([text], prompt_name=prompt_name)[0].tolist()
        
        # Save to caches
        self._memory_cache[cache_key] = embedding
        self._save_to_disk_cache(cache_key, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str], prompt_name: str) -> List[List[float]]:
        """
        Embed multiple texts efficiently using batching.
        
        Args:
            texts: List of texts to embed
            prompt_name: Type of embedding prompt
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, prompt_name=prompt_name)
            results.extend(batch_embeddings.tolist())
        
        return results
    
    def embed(self, texts: Union[str, List[str]], prompt_name: str) -> Union[List[float], List[List[float]]]:
        """
        Main embedding method that handles both single and batch inputs efficiently.
        
        Args:
            texts: Single text string or list of texts
            prompt_name: Type of embedding prompt
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        if isinstance(texts, str):
            # Single text - use cached method
            return self.embed_single_cached(texts, prompt_name)
        elif isinstance(texts, list):
            # Multiple texts - use batch processing
            return self.embed_batch(texts, prompt_name)
        else:
            raise ValueError("Input must be a string or list of strings")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self._cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Clear disk cache
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))


# Backward compatibility - keep the old class name
class EmbeddingModel(OptimizedEmbeddingModel):
    """Backward compatibility wrapper for the old EmbeddingModel class."""
    pass
