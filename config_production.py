#!/usr/bin/env python3
"""
Production Configuration for Emma's Digital Limbic System
Safety limits, performance constraints, and security settings
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductionConfig:
    """Production-ready configuration with safety limits"""
    
    # Memory Management
    MAX_MEMORY_SIZE: int = 100_000_000  # 100MB
    MAX_MEMORIES_PER_SESSION: int = 1000
    MEMORY_CONSOLIDATION_THRESHOLD: float = 0.9  # Trigger at 90% capacity
    MEMORY_PRUNING_THRESHOLD: float = 0.01
    
    # Session Management
    MAX_SESSION_DURATION: int = 7200  # 2 hours in seconds
    MAX_MESSAGES_PER_SESSION: int = 500
    MAX_CONCURRENT_SESSIONS: int = 100
    SESSION_IDLE_TIMEOUT: int = 1800  # 30 minutes
    
    # State Safety
    STATE_NORM_LIMIT: float = 10.0
    STATE_CHANGE_RATE_LIMIT: float = 2.0
    STATE_VALIDATION_ENABLED: bool = True
    STATE_RECOVERY_ENABLED: bool = True
    
    # API Integration
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_TIMEOUT: float = 30.0
    GEMINI_RATE_LIMIT_RPM: int = 60  # Requests per minute
    GEMINI_CIRCUIT_BREAKER_THRESHOLD: int = 5
    GEMINI_CIRCUIT_BREAKER_TIMEOUT: float = 300.0  # 5 minutes
    
    # Input Validation
    MAX_INPUT_LENGTH: int = 5000
    INPUT_SANITIZATION_ENABLED: bool = True
    PROFANITY_FILTER_ENABLED: bool = True
    
    # Performance Limits
    MAX_PROCESSING_TIME: float = 30.0  # seconds
    MAX_CPU_USAGE: float = 80.0  # percent
    MAX_MEMORY_USAGE: int = 1000  # MB
    
    # Security
    SESSION_AUTHENTICATION_REQUIRED: bool = False  # Set to True for production
    RATE_LIMITING_ENABLED: bool = True
    REQUEST_LOGGING_ENABLED: bool = True
    ERROR_DETAILS_IN_RESPONSE: bool = False  # Hide details in production
    
    # Monitoring
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    METRICS_RETENTION_HOURS: int = 24
    ALERT_THRESHOLDS_ENABLED: bool = True
    
    # Database/Storage
    CONVERSATION_HISTORY_LIMIT: int = 1000  # Messages per session
    STATE_PERSISTENCE_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 6
    
    # Experimental Features
    ENABLE_EXPERIMENTAL_FEATURES: bool = False
    DEBUG_MODE: bool = False
    VERBOSE_LOGGING: bool = False

class ConfigManager:
    """Manages configuration with environment variable overrides"""
    
    def __init__(self):
        self.config = ProductionConfig()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Memory settings
        self.config.MAX_MEMORY_SIZE = int(os.getenv('EMMA_MAX_MEMORY_SIZE', self.config.MAX_MEMORY_SIZE))
        self.config.MAX_MEMORIES_PER_SESSION = int(os.getenv('EMMA_MAX_MEMORIES_PER_SESSION', self.config.MAX_MEMORIES_PER_SESSION))
        
        # Session settings
        self.config.MAX_SESSION_DURATION = int(os.getenv('EMMA_MAX_SESSION_DURATION', self.config.MAX_SESSION_DURATION))
        self.config.MAX_MESSAGES_PER_SESSION = int(os.getenv('EMMA_MAX_MESSAGES_PER_SESSION', self.config.MAX_MESSAGES_PER_SESSION))
        self.config.MAX_CONCURRENT_SESSIONS = int(os.getenv('EMMA_MAX_CONCURRENT_SESSIONS', self.config.MAX_CONCURRENT_SESSIONS))
        
        # Security settings
        self.config.SESSION_AUTHENTICATION_REQUIRED = os.getenv('EMMA_REQUIRE_AUTH', 'false').lower() == 'true'
        self.config.RATE_LIMITING_ENABLED = os.getenv('EMMA_RATE_LIMITING', 'true').lower() == 'true'
        
        # Debug settings
        self.config.DEBUG_MODE = os.getenv('EMMA_DEBUG', 'false').lower() == 'true'
        self.config.VERBOSE_LOGGING = os.getenv('EMMA_VERBOSE_LOGGING', 'false').lower() == 'true'
    
    def get_config(self) -> ProductionConfig:
        """Get the current configuration"""
        return self.config
    
    def validate_config(self) -> bool:
        """Validate configuration for production readiness"""
        issues = []
        
        # Check critical limits
        if self.config.MAX_MEMORY_SIZE > 500_000_000:  # 500MB
            issues.append("Memory limit too high for production")
        
        if self.config.MAX_CONCURRENT_SESSIONS > 500:
            issues.append("Concurrent session limit too high")
        
        if self.config.STATE_NORM_LIMIT > 20.0:
            issues.append("State norm limit too high - risk of instability")
        
        # Check security settings for production
        if not self.config.RATE_LIMITING_ENABLED:
            issues.append("Rate limiting should be enabled in production")
        
        if self.config.ERROR_DETAILS_IN_RESPONSE:
            issues.append("Error details should be hidden in production")
        
        if self.config.DEBUG_MODE:
            issues.append("Debug mode should be disabled in production")
        
        if issues:
            print("Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True

# Input sanitization functions
def sanitize_user_input(text: str, config: ProductionConfig) -> str:
    """Sanitize user input for security"""
    if not config.INPUT_SANITIZATION_ENABLED:
        return text
    
    # Length check
    if len(text) > config.MAX_INPUT_LENGTH:
        text = text[:config.MAX_INPUT_LENGTH]
    
    # Remove potentially dangerous content
    import re
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script injection attempts
    text = re.sub(r'(javascript:|data:|vbscript:)', '', text, flags=re.IGNORECASE)
    
    # Remove SQL injection attempts
    text = re.sub(r'(union\s+select|drop\s+table|delete\s+from)', '', text, flags=re.IGNORECASE)
    
    # Basic profanity filter (if enabled)
    if config.PROFANITY_FILTER_ENABLED:
        # This would integrate with a proper profanity filter library
        # For now, just a placeholder
        pass
    
    return text.strip()

def validate_session_limits(session_data: dict, config: ProductionConfig) -> tuple[bool, str]:
    """Validate session against configured limits"""
    import time
    
    # Check session duration
    session_age = time.time() - session_data.get('created_at', time.time())
    if session_age > config.MAX_SESSION_DURATION:
        return False, "Session duration limit exceeded"
    
    # Check message count
    message_count = session_data.get('message_count', 0)
    if message_count > config.MAX_MESSAGES_PER_SESSION:
        return False, "Message limit exceeded"
    
    # Check memory usage
    memory_count = len(session_data.get('memories', []))
    if memory_count > config.MAX_MEMORIES_PER_SESSION:
        return False, "Memory limit exceeded"
    
    return True, "Session within limits"

# Global configuration instance
config_manager = ConfigManager()
production_config = config_manager.get_config()

# Validate configuration on import
if not config_manager.validate_config():
    import warnings
    warnings.warn("Production configuration validation failed. Review settings before deployment.")

def get_production_config() -> ProductionConfig:
    """Get the production configuration instance"""
    return production_config

if __name__ == "__main__":
    # Configuration validation script
    print("Emma Production Configuration Validation")
    print("=" * 50)
    
    config = get_production_config()
    
    print(f"Max Memory Size: {config.MAX_MEMORY_SIZE:,} bytes")
    print(f"Max Concurrent Sessions: {config.MAX_CONCURRENT_SESSIONS}")
    print(f"Max Session Duration: {config.MAX_SESSION_DURATION} seconds")
    print(f"State Norm Limit: {config.STATE_NORM_LIMIT}")
    print(f"Rate Limiting: {'Enabled' if config.RATE_LIMITING_ENABLED else 'Disabled'}")
    print(f"Debug Mode: {'Enabled' if config.DEBUG_MODE else 'Disabled'}")
    
    print("\nValidation Result:", "PASSED" if config_manager.validate_config() else "FAILED")
