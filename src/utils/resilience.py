import time
import asyncio
import logging
from typing import Any, Callable, Optional, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import random
import json
import os

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    monitor_interval: float = 10.0  # seconds

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_backoff: bool = True
    jitter: bool = True

@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    enable_fallback: bool = True
    fallback_response: str = "I'm experiencing technical difficulties. Please try again later."
    cache_fallback_responses: bool = True
    fallback_cache_ttl: int = 300  # seconds

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for model resilience.
    
    Prevents cascading failures by temporarily stopping requests to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.successful_requests = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.total_requests += 1
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._set_half_open()
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.last_success_time = time.time()
        self.successful_requests += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self._set_closed()
            logger.info(f"Circuit breaker '{self.name}' recovered and is now CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.failed_requests += 1
        
        if self.failure_count >= self.config.failure_threshold:
            self._set_open()
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _set_open(self):
        """Set circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
    
    def _set_half_open(self):
        """Set circuit breaker to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
    
    def _set_closed(self):
        """Set circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
    
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time
        }

class RetryHandler:
    """
    Retry logic with exponential backoff and jitter.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {e}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** attempt)
        else:
            delay = self.config.base_delay
        
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)  # Add 50% jitter
        
        return delay

class FallbackManager:
    """
    Manages fallback responses and cached fallback mechanisms.
    """
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.fallback_cache = {}
        self.fallback_cache_timestamps = {}
    
    def get_fallback_response(self, query: str, context: Optional[Dict] = None) -> str:
        """Get appropriate fallback response."""
        if not self.config.enable_fallback:
            raise FallbackDisabledError("Fallback is disabled")
        
        # Check cache first
        if self.config.cache_fallback_responses:
            cached_response = self._get_cached_fallback(query)
            if cached_response:
                return cached_response
        
        # Generate fallback response
        response = self._generate_fallback_response(query, context)
        
        # Cache the response
        if self.config.cache_fallback_responses:
            self._cache_fallback_response(query, response)
        
        return response
    
    def _generate_fallback_response(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate context-aware fallback response."""
        # Simple keyword-based fallback responses
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["admission", "requirement", "apply"]):
            return "I'm having trouble accessing the admission information right now. Please check the university website or contact the admissions office directly for the most current requirements."
        
        elif any(word in query_lower for word in ["cost", "fee", "tuition", "price"]):
            return "I'm unable to retrieve cost information at the moment. Please visit the university's financial aid page or contact the bursar's office for current tuition and fee information."
        
        elif any(word in query_lower for word in ["deadline", "date", "when"]):
            return "I'm experiencing technical difficulties accessing deadline information. Please check the university's academic calendar or contact the relevant department for current deadlines."
        
        elif any(word in query_lower for word in ["scholarship", "financial aid", "funding"]):
            return "I'm having trouble accessing scholarship information. Please visit the university's financial aid website or contact the financial aid office for current scholarship opportunities."
        
        else:
            return self.config.fallback_response
    
    def _get_cached_fallback(self, query: str) -> Optional[str]:
        """Get cached fallback response if available and not expired."""
        if query in self.fallback_cache:
            timestamp = self.fallback_cache_timestamps.get(query, 0)
            if time.time() - timestamp < self.config.fallback_cache_ttl:
                return self.fallback_cache[query]
            else:
                # Remove expired cache entry
                del self.fallback_cache[query]
                del self.fallback_cache_timestamps[query]
        return None
    
    def _cache_fallback_response(self, query: str, response: str):
        """Cache fallback response."""
        self.fallback_cache[query] = response
        self.fallback_cache_timestamps[query] = time.time()
    
    def clear_cache(self):
        """Clear fallback cache."""
        self.fallback_cache.clear()
        self.fallback_cache_timestamps.clear()

class ResilienceManager:
    """
    Main resilience manager that coordinates circuit breakers, retry logic, and fallbacks.
    """
    
    def __init__(
        self,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None
    ):
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.retry_config = retry_config or RetryConfig()
        self.fallback_config = fallback_config or FallbackConfig()
        
        # Initialize components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler(self.retry_config)
        self.fallback_manager = FallbackManager(self.fallback_config)
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.fallback_calls = 0
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, self.circuit_breaker_config)
        return self.circuit_breakers[name]
    
    def call_with_resilience(
        self,
        service_name: str,
        func: Callable,
        *args,
        query: str = "",
        context: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with full resilience protection.
        
        Args:
            service_name: Name of the service for circuit breaker
            func: Function to execute
            *args: Function arguments
            query: User query for fallback generation
            context: Additional context for fallback
            **kwargs: Function keyword arguments
        """
        self.total_calls += 1
        
        try:
            # Get circuit breaker for this service
            circuit_breaker = self.get_circuit_breaker(service_name)
            
            # Execute with circuit breaker and retry protection
            def protected_call():
                return circuit_breaker.call(func, *args, **kwargs)
            
            result = self.retry_handler.call(protected_call)
            self.successful_calls += 1
            return result
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Service '{service_name}' failed after retries: {e}")
            
            # Try fallback
            try:
                fallback_response = self.fallback_manager.get_fallback_response(query, context)
                self.fallback_calls += 1
                logger.info(f"Using fallback response for service '{service_name}'")
                return fallback_response
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for service '{service_name}': {fallback_error}")
                raise ResilienceError(f"Service '{service_name}' failed and fallback unavailable: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive resilience statistics."""
        circuit_breaker_stats = {
            name: cb.get_stats() for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "fallback_calls": self.fallback_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "fallback_rate": self.fallback_calls / self.total_calls if self.total_calls > 0 else 0,
            "circuit_breakers": circuit_breaker_stats,
            "config": {
                "circuit_breaker": {
                    "failure_threshold": self.circuit_breaker_config.failure_threshold,
                    "recovery_timeout": self.circuit_breaker_config.recovery_timeout
                },
                "retry": {
                    "max_attempts": self.retry_config.max_attempts,
                    "base_delay": self.retry_config.base_delay
                },
                "fallback": {
                    "enabled": self.fallback_config.enable_fallback,
                    "cache_enabled": self.fallback_config.cache_fallback_responses
                }
            }
        }

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class FallbackDisabledError(Exception):
    """Raised when fallback is disabled."""
    pass

class ResilienceError(Exception):
    """Raised when all resilience mechanisms fail."""
    pass

# Decorator for easy resilience application
def resilient(service_name: str, query_param: str = "query"):
    """
    Decorator to add resilience to functions.
    
    Args:
        service_name: Name of the service for circuit breaker
        query_param: Name of the parameter containing the user query
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get resilience manager (you might want to make this configurable)
            resilience_manager = get_resilience_manager()
            
            # Extract query from kwargs
            query = kwargs.get(query_param, "")
            context = kwargs.get("context", None)
            
            return resilience_manager.call_with_resilience(
                service_name=service_name,
                func=func,
                *args,
                query=query,
                context=context,
                **kwargs
            )
        return wrapper
    return decorator

# Global resilience manager instance
_global_resilience_manager = None

def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager instance."""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager()
    return _global_resilience_manager

def configure_resilience(
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    fallback_config: Optional[FallbackConfig] = None
):
    """Configure the global resilience manager."""
    global _global_resilience_manager
    _global_resilience_manager = ResilienceManager(
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config,
        fallback_config=fallback_config
    ) 