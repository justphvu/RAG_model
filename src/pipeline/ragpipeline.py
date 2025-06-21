import json
import time
from typing import List, Dict, Optional, Union
from src.config import Constants
from src.models.conversation import ConversationManager, Message

class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with conversation support, caching, and performance optimizations.
    
    Features:
    - Conversation history management
    - Context-aware responses
    - Performance monitoring
    - Caching for repeated queries
    """
    
    def __init__(
        self,
        retriever,
        llm,
        conversation_manager: Optional[ConversationManager] = None,
        device: str = Constants.DEVICE,
        enable_query_cache: bool = True,
        max_context_tokens: int = 2000
    ):
        self.retriever = retriever
        self.llm = llm
        self.device = device
        self.enable_query_cache = enable_query_cache
        self.max_context_tokens = max_context_tokens
        
        # Initialize conversation manager if not provided
        if conversation_manager is None:
            self.conversation_manager = ConversationManager()
        else:
            self.conversation_manager = conversation_manager
        
        # Query cache for repeated questions
        self._query_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Performance monitoring
        self._performance_stats = {
            "total_queries": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
    
    def _generate_cache_key(self, query: str, user_id: str, context_messages: List[Message]) -> str:
        """Generate a cache key for query + user context."""
        import hashlib
        context_str = "".join([msg.content for msg in context_messages])
        content = f"{query}:{user_id}:{context_str}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.enable_query_cache:
            return None
        
        if cache_key in self._query_cache:
            self._cache_stats["hits"] += 1
            return self._query_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        return None
    
    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache a response."""
        if not self.enable_query_cache:
            return
        
        # Simple LRU-like cache with size limit
        if len(self._query_cache) > 1000:
            # Remove oldest entry (simple implementation)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = response

    def generate_prompt(
        self,
        question: str,
        answers: List[dict],
        conversation_history: List[Message] = None,
        system_prompt: str = Constants.DEFAULT_SYSTEM_PROMPT
    ) -> List[Dict]:
        """
        Enhanced prompt generation with conversation history support.

        Args:
            answers: List of retrieved documents
            question: Current user question
            conversation_history: Previous conversation messages
            system_prompt: System prompt for the model

        Returns:
            List of messages in chat format
        """
        # Format retrieved documents
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
                continue

            if not text.strip():
                continue

            formatted_docs.append({
                "doc_id": i,
                "title": title,
                "content": text
            })

        # Build context from documents
        context_text = f"""
        {json.dumps(formatted_docs, ensure_ascii=False)}
        """
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join([
                f"{msg.role}: {msg.content}" 
                for msg in conversation_history[-3:]  # Last 3 messages for context
            ])
        
        # Combine everything into the prompt
        if conversation_context:
            full_query = f"""
            Previous conversation:
            {conversation_context}
            
            Retrieved documents:
            {context_text}
            
            Current question: {question}
            
            Please answer the current question based on the retrieved documents and conversation context.
            """
        else:
            full_query = f"""
            {context_text}
            На основе выше документов ответь на следующий вопрос:
            {question}
            """
        
        # Build chat format
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_query},
        ]
        
        return prompt

    def generate_answer(
        self, 
        query: str, 
        user_id: str = "default",
        top_k: int = 10,
        use_conversation_history: bool = True
    ) -> str:
        """
        Enhanced answer generation with conversation support and caching.

        Args:
            query: User's question
            user_id: Unique user identifier
            top_k: Number of top documents to retrieve
            use_conversation_history: Whether to include conversation history

        Returns:
            Generated answer
        """
        start_time = time.time()
        
        try:
            # Get conversation history if enabled
            conversation_history = []
            if use_conversation_history:
                conversation_history = self.conversation_manager.get_context_window(
                    user_id, self.max_context_tokens
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(query, user_id, conversation_history)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                # Still add to conversation history
                self.conversation_manager.add_user_message(user_id, query)
                self.conversation_manager.add_assistant_message(user_id, cached_response)
                return cached_response
            
            # Add user message to conversation
            self.conversation_manager.add_user_message(user_id, query)
            
            # Step 1: Generate alternative queries
            alt_queries = self.retriever.generate_alternative_queries(query)

            # Step 2: Retrieve relevant child docs
            child_docs = self.retriever.query_vectorstore(alt_queries)

            # Step 3: Get parent docs
            parent_docs = self.retriever.retrieve_parent_docs(child_docs)

            # Step 4: Rerank
            reranked_docs = self.retriever.rerank(query, parent_docs, top_k=top_k)

            # Step 5: Build enhanced prompt
            prompt = self.generate_prompt(
                query, 
                reranked_docs, 
                conversation_history if use_conversation_history else None
            )

            # Step 6: Generate answer
            answer = self.llm.generate(prompt)
            
            # Cache the response
            self._cache_response(cache_key, answer)
            
            # Add assistant response to conversation
            self.conversation_manager.add_assistant_message(user_id, answer)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            return answer
            
        except Exception as e:
            # Log error and return fallback response
            error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
            self.conversation_manager.add_assistant_message(user_id, error_msg)
            return error_msg
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics."""
        self._performance_stats["total_queries"] += 1
        self._performance_stats["total_processing_time"] += processing_time
        self._performance_stats["avg_processing_time"] = (
            self._performance_stats["total_processing_time"] / 
            self._performance_stats["total_queries"]
        )
    
    def get_conversation_history(self, user_id: str, max_messages: int = 10) -> List[Message]:
        """Get conversation history for a user."""
        return self.conversation_manager.get_conversation_history(user_id, max_messages)
    
    def clear_conversation(self, user_id: str) -> None:
        """Clear conversation history for a user."""
        self.conversation_manager.clear_conversation(user_id)
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self._cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._query_cache)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self._performance_stats.copy()
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation manager statistics."""
        return self.conversation_manager.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._query_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def cleanup_old_conversations(self) -> int:
        """Clean up old conversations."""
        return self.conversation_manager.cleanup_old_conversations()


# Backward compatibility - keep the old class name
class RAGPipeline(EnhancedRAGPipeline):
    """Backward compatibility wrapper for the old RAGPipeline class."""
    
    def generate_answer(self, query: str, top_k: int = 10) -> str:
        """Backward compatible method without conversation support."""
        return super().generate_answer(query, user_id="default", top_k=top_k, use_conversation_history=False)
