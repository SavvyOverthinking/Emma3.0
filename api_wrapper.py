#!/usr/bin/env python3
"""
Resilient API Wrapper for Emma's Gemini Integration
Handles retries, rate limiting, fallbacks, and error recovery
"""

import asyncio
import time
import logging
import random
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    AUTH = "authentication"
    MODEL = "model_error"
    QUOTA = "quota_exceeded"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class APICall:
    """Represents an API call with retry context"""
    prompt: str
    context: Dict[str, Any]
    attempt: int = 0
    last_error: Optional[str] = None
    start_time: float = 0.0
    total_time: float = 0.0

class RateLimiter:
    """Smart rate limiter with adaptive delays"""
    
    def __init__(self, calls_per_minute: int = 60, burst_limit: int = 10):
        self.calls_per_minute = calls_per_minute
        self.burst_limit = burst_limit
        self.call_times = []
        self.burst_count = 0
        self.last_reset = time.time()
    
    async def acquire(self) -> float:
        """Acquire rate limit token, returns delay time"""
        current_time = time.time()
        
        # Reset burst counter every minute
        if current_time - self.last_reset > 60:
            self.burst_count = 0
            self.last_reset = current_time
            self.call_times = [t for t in self.call_times if current_time - t < 60]
        
        # Check burst limit
        if self.burst_count >= self.burst_limit:
            delay = 60 - (current_time - self.last_reset)
            if delay > 0:
                logger.info(f"Rate limiter: burst limit hit, waiting {delay:.1f}s")
                await asyncio.sleep(delay)
                self.burst_count = 0
                self.last_reset = time.time()
        
        # Check per-minute limit
        recent_calls = len([t for t in self.call_times if current_time - t < 60])
        if recent_calls >= self.calls_per_minute:
            oldest_call = min(self.call_times)
            delay = 60 - (current_time - oldest_call) + 1
            logger.info(f"Rate limiter: per-minute limit hit, waiting {delay:.1f}s")
            await asyncio.sleep(delay)
            current_time = time.time()
        
        # Record call
        self.call_times.append(current_time)
        self.burst_count += 1
        
        return 0.0

class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
    
    def can_proceed(self) -> bool:
        """Check if API calls should proceed"""
        current_time = time.time()
        
        if self.state == "open":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "half_open"
                logger.info("Circuit breaker moving to half-open state")
                return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful API call"""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker reset to closed state")
    
    def record_failure(self):
        """Record failed API call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class ResilientGeminiAPI:
    """Resilient Gemini API wrapper with comprehensive error handling"""
    
    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 1.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Initialize Gemini client
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize resilience components
        self.rate_limiter = RateLimiter(calls_per_minute=60, burst_limit=10)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=300.0)
        
        # Performance tracking
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retried_calls': 0,
            'fallback_calls': 0,
            'avg_response_time': 0.0,
            'error_distribution': {},
            'last_success_time': time.time(),
            'circuit_breaker_opens': 0
        }
        
        logger.info("ResilientGeminiAPI initialized with adaptive error handling")
    
    def _classify_error(self, error: Exception) -> APIErrorType:
        """Classify API errors for appropriate handling"""
        error_str = str(error).lower()
        
        if isinstance(error, google_exceptions.TooManyRequests) or "rate limit" in error_str:
            return APIErrorType.RATE_LIMIT
        elif isinstance(error, google_exceptions.Unauthenticated) or "auth" in error_str:
            return APIErrorType.AUTH
        elif isinstance(error, google_exceptions.DeadlineExceeded) or "timeout" in error_str:
            return APIErrorType.TIMEOUT
        elif "quota" in error_str or "exceeded" in error_str:
            return APIErrorType.QUOTA
        elif "network" in error_str or "connection" in error_str:
            return APIErrorType.NETWORK
        elif "model" in error_str or "not found" in error_str:
            return APIErrorType.MODEL
        else:
            return APIErrorType.UNKNOWN
    
    def _calculate_backoff_delay(self, attempt: int, error_type: APIErrorType) -> float:
        """Calculate adaptive backoff delay based on error type"""
        base_delay = self.base_delay
        
        # Adjust base delay by error type
        error_multipliers = {
            APIErrorType.RATE_LIMIT: 2.0,
            APIErrorType.NETWORK: 1.5,
            APIErrorType.TIMEOUT: 1.2,
            APIErrorType.QUOTA: 3.0,
            APIErrorType.MODEL: 0.5,
            APIErrorType.AUTH: 0.1,  # Don't wait long for auth errors
            APIErrorType.UNKNOWN: 1.0
        }
        
        multiplier = error_multipliers.get(error_type, 1.0)
        
        # Exponential backoff with jitter
        delay = base_delay * multiplier * (2 ** attempt)
        jitter = delay * 0.1 * random.random()
        
        return min(delay + jitter, 60.0)  # Cap at 60 seconds
    
    async def generate_with_resilience(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response with full resilience features"""
        self.stats['total_calls'] += 1
        api_call = APICall(prompt=prompt, context=context or {}, start_time=time.time())
        
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open, using fallback")
            return await self._generate_fallback_response(api_call)
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Retry loop
        for attempt in range(self.max_retries + 1):
            api_call.attempt = attempt + 1
            
            try:
                # Generate response
                response = self.model.generate_content(prompt)
                
                # Success
                self.circuit_breaker.record_success()
                self.stats['successful_calls'] += 1
                self.stats['last_success_time'] = time.time()
                
                # Update performance stats
                api_call.total_time = time.time() - api_call.start_time
                self._update_performance_stats(api_call.total_time)
                
                return response.text
                
            except Exception as error:
                error_type = self._classify_error(error)
                api_call.last_error = str(error)
                
                # Update error stats
                self.stats['error_distribution'][error_type.value] = (
                    self.stats['error_distribution'].get(error_type.value, 0) + 1
                )
                
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {error_type.value} - {error}")
                
                # Don't retry certain errors
                if error_type in [APIErrorType.AUTH, APIErrorType.MODEL]:
                    break
                
                # Last attempt - record failure and use fallback
                if attempt == self.max_retries:
                    self.circuit_breaker.record_failure()
                    self.stats['failed_calls'] += 1
                    break
                
                # Calculate backoff delay
                delay = self._calculate_backoff_delay(attempt, error_type)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                self.stats['retried_calls'] += 1
        
        # All retries failed, use fallback
        return await self._generate_fallback_response(api_call)
    
    async def _generate_fallback_response(self, api_call: APICall) -> str:
        """Generate fallback response when API fails"""
        self.stats['fallback_calls'] += 1
        
        # Analyze context to generate appropriate fallback
        context = api_call.context
        prompt = api_call.prompt.lower()
        
        # Emotional state-aware fallbacks
        if context.get('emotional_state'):
            emotions = context['emotional_state']
            if emotions.get('uncertainty', 0) > 0.6:
                fallbacks = [
                    "i'm feeling a bit scattered right now, give me a moment to gather my thoughts",
                    "my mind's all over the place... what were we talking about?",
                    "sorry, i'm having trouble focusing. can you help me understand what you meant?"
                ]
            elif emotions.get('fatigue', 0) > 0.7:
                fallbacks = [
                    "i'm feeling pretty tired right now, might need a sec to process that",
                    "my brain's a bit foggy today, can you say that again?",
                    "sorry, i'm running low on energy. what was the question?"
                ]
            else:
                fallbacks = [
                    "hmm, i lost my train of thought. can you remind me what we were talking about?",
                    "sorry, something just went fuzzy in my head there",
                    "i need a moment to collect my thoughts"
                ]
        else:
            # Generic fallbacks
            fallbacks = [
                "i'm having a bit of trouble thinking right now. can you try asking that again?",
                "sorry, my mind wandered there for a second. what were you saying?",
                "hmm, i seem to have lost focus. could you repeat that?",
                "i need a moment to gather my thoughts. give me a sec?"
            ]
        
        response = random.choice(fallbacks)
        logger.info(f"Generated fallback response: {response[:50]}...")
        
        return response
    
    def _update_performance_stats(self, response_time: float):
        """Update performance statistics"""
        if self.stats['successful_calls'] == 1:
            self.stats['avg_response_time'] = response_time
        else:
            # Running average
            alpha = 0.1  # Smoothing factor
            self.stats['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.stats['avg_response_time']
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive API health status"""
        current_time = time.time()
        success_rate = (
            self.stats['successful_calls'] / max(self.stats['total_calls'], 1)
        )
        
        # Determine health status
        if self.circuit_breaker.state == "open":
            health = "critical"
        elif success_rate < 0.5:
            health = "unhealthy"
        elif success_rate < 0.8:
            health = "degraded"
        else:
            health = "healthy"
        
        return {
            'health': health,
            'circuit_breaker_state': self.circuit_breaker.state,
            'success_rate': success_rate,
            'total_calls': self.stats['total_calls'],
            'successful_calls': self.stats['successful_calls'],
            'failed_calls': self.stats['failed_calls'],
            'fallback_calls': self.stats['fallback_calls'],
            'avg_response_time': self.stats['avg_response_time'],
            'time_since_last_success': current_time - self.stats['last_success_time'],
            'error_distribution': self.stats['error_distribution'],
            'recommendations': self._get_health_recommendations(health, success_rate)
        }
    
    def _get_health_recommendations(self, health: str, success_rate: float) -> List[str]:
        """Get recommendations based on health status"""
        recommendations = []
        
        if health == "critical":
            recommendations.append("API completely down - check authentication and network")
            recommendations.append("Consider switching to offline mode")
        elif health == "unhealthy":
            recommendations.append(f"Low success rate ({success_rate:.1%}) - investigate API issues")
            recommendations.append("Increase fallback usage")
        elif health == "degraded":
            recommendations.append("Moderate API issues - monitor closely")
            recommendations.append("Consider reducing request rate")
        
        # Error-specific recommendations
        errors = self.stats['error_distribution']
        if errors.get('rate_limit', 0) > 5:
            recommendations.append("Frequent rate limiting - reduce request frequency")
        if errors.get('timeout', 0) > 3:
            recommendations.append("Timeout issues - check network connectivity")
        if errors.get('quota', 0) > 0:
            recommendations.append("Quota exceeded - check API usage limits")
        
        return recommendations
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retried_calls': 0,
            'fallback_calls': 0,
            'avg_response_time': 0.0,
            'error_distribution': {},
            'last_success_time': time.time(),
            'circuit_breaker_opens': 0
        }
        logger.info("API wrapper stats reset")

# Synchronous wrapper for compatibility
class SyncResilientGeminiAPI:
    """Synchronous wrapper around the async API"""
    
    def __init__(self, api_key: str, max_retries: int = 3):
        self.async_api = ResilientGeminiAPI(api_key, max_retries)
    
    def generate_with_resilience(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Synchronous version of resilient generation"""
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.async_api.generate_with_resilience(prompt, context)
            )
        finally:
            loop.close()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status synchronously"""
        return self.async_api.get_health_status()

async def test_resilient_api():
    """Test the resilient API wrapper"""
    print("Testing Resilient Gemini API")
    print("=" * 50)
    
    api_key = "test_key_for_demo"  # Would use real key in production
    api = ResilientGeminiAPI(api_key, max_retries=2)
    
    # Test normal operation (will fail with test key, demonstrating fallback)
    test_prompts = [
        "Tell me about yourself",
        "What's your favorite color?",
        "How are you feeling today?"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        try:
            response = await api.generate_with_resilience(prompt, {
                'emotional_state': {'uncertainty': 0.3, 'fatigue': 0.5}
            })
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Show health status
    health = api.get_health_status()
    print(f"\nAPI Health Status:")
    print(f"  Health: {health['health']}")
    print(f"  Success rate: {health['success_rate']:.1%}")
    print(f"  Total calls: {health['total_calls']}")
    print(f"  Fallback calls: {health['fallback_calls']}")

if __name__ == "__main__":
    asyncio.run(test_resilient_api())
