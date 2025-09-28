#!/usr/bin/env python3
"""
Session Manager for Emma's Digital Limbic System
Handles session lifecycle, limits, and resource management
"""

import time
import uuid
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class SessionLimits:
    """Configuration for session limits"""
    max_duration_hours: float = 2.0
    max_messages: int = 500
    max_memory_mb: int = 500
    idle_timeout_minutes: float = 30.0
    max_concurrent_sessions: int = 10
    max_processing_time_seconds: float = 30.0

@dataclass
class SessionInfo:
    """Information about an active session"""
    session_id: str
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    processing_time_total: float = 0.0
    memory_usage_mb: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None

class SessionManager:
    """Advanced session management with limits and monitoring"""
    
    def __init__(self, limits: SessionLimits = None):
        self.limits = limits or SessionLimits()
        self.sessions: Dict[str, SessionInfo] = {}
        self.cleanup_thread = None
        self.running = True
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self.start_cleanup_thread()
        
        logger.info(f"SessionManager initialized with limits: {self.limits}")
    
    def create_session(self, user_id: Optional[str] = None) -> Tuple[str, bool]:
        """Create a new session"""
        with self._lock:
            # Check concurrent session limit
            active_sessions = sum(1 for s in self.sessions.values() if s.status == SessionStatus.ACTIVE)
            if active_sessions >= self.limits.max_concurrent_sessions:
                logger.warning(f"Max concurrent sessions reached: {active_sessions}")
                return None, False
            
            session_id = str(uuid.uuid4())
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                metadata={'created_at': time.time()}
            )
            
            self.sessions[session_id] = session_info
            logger.info(f"Created session {session_id} for user {user_id}")
            
            return session_id, True
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate if session can continue processing"""
        with self._lock:
            if session_id not in self.sessions:
                return False, "Session not found"
            
            session = self.sessions[session_id]
            
            # Check if session is terminated
            if session.status in [SessionStatus.EXPIRED, SessionStatus.TERMINATED, SessionStatus.ERROR]:
                return False, f"Session status: {session.status.value}"
            
            # Check duration limit
            duration_hours = (time.time() - session.start_time) / 3600
            if duration_hours > self.limits.max_duration_hours:
                session.status = SessionStatus.EXPIRED
                logger.info(f"Session {session_id} expired due to duration: {duration_hours:.2f}h")
                return False, f"Session expired (duration: {duration_hours:.2f}h)"
            
            # Check message limit
            if session.message_count >= self.limits.max_messages:
                session.status = SessionStatus.EXPIRED
                logger.info(f"Session {session_id} expired due to message count: {session.message_count}")
                return False, f"Session expired (message limit: {session.message_count})"
            
            # Check memory limit
            if session.memory_usage_mb > self.limits.max_memory_mb:
                session.status = SessionStatus.EXPIRED
                logger.warning(f"Session {session_id} expired due to memory usage: {session.memory_usage_mb}MB")
                return False, f"Session expired (memory limit: {session.memory_usage_mb}MB)"
            
            # Check idle timeout
            idle_time_minutes = (time.time() - session.last_activity) / 60
            if idle_time_minutes > self.limits.idle_timeout_minutes:
                session.status = SessionStatus.IDLE
                logger.info(f"Session {session_id} idle for {idle_time_minutes:.1f} minutes")
                return False, f"Session idle (timeout: {idle_time_minutes:.1f}m)"
            
            return True, None
    
    def update_session_activity(self, session_id: str, message_processed: bool = False, 
                              processing_time: float = 0.0, memory_usage_mb: float = 0.0,
                              error: Optional[str] = None):
        """Update session activity and metrics"""
        with self._lock:
            if session_id not in self.sessions:
                logger.warning(f"Attempted to update non-existent session: {session_id}")
                return
            
            session = self.sessions[session_id]
            session.last_activity = time.time()
            
            if message_processed:
                session.message_count += 1
            
            if processing_time > 0:
                session.processing_time_total += processing_time
                
                # Check processing time limit
                if processing_time > self.limits.max_processing_time_seconds:
                    logger.warning(f"Session {session_id} exceeded processing time limit: {processing_time:.2f}s")
            
            if memory_usage_mb > 0:
                session.memory_usage_mb = memory_usage_mb
            
            if error:
                session.error_count += 1
                session.last_error = error
                
                # Terminate session if too many errors
                if session.error_count >= 5:
                    session.status = SessionStatus.ERROR
                    logger.error(f"Session {session_id} terminated due to excessive errors: {session.error_count}")
            
            # Reactivate idle sessions
            if session.status == SessionStatus.IDLE:
                session.status = SessionStatus.ACTIVE
                logger.info(f"Session {session_id} reactivated from idle")
    
    def terminate_session(self, session_id: str, reason: str = "user_requested"):
        """Manually terminate a session"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.status = SessionStatus.TERMINATED
                session.metadata['termination_reason'] = reason
                session.metadata['terminated_at'] = time.time()
                logger.info(f"Session {session_id} terminated: {reason}")
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        with self._lock:
            return self.sessions.get(session_id)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session statistics"""
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            current_time = time.time()
            
            duration_minutes = (current_time - session.start_time) / 60
            idle_minutes = (current_time - session.last_activity) / 60
            avg_processing_time = (session.processing_time_total / max(session.message_count, 1))
            
            return {
                'session_id': session_id,
                'status': session.status.value,
                'duration_minutes': duration_minutes,
                'idle_minutes': idle_minutes,
                'message_count': session.message_count,
                'error_count': session.error_count,
                'avg_processing_time': avg_processing_time,
                'total_processing_time': session.processing_time_total,
                'memory_usage_mb': session.memory_usage_mb,
                'utilization': {
                    'duration_pct': (duration_minutes / 60) / self.limits.max_duration_hours * 100,
                    'message_pct': session.message_count / self.limits.max_messages * 100,
                    'memory_pct': session.memory_usage_mb / self.limits.max_memory_mb * 100
                },
                'limits_status': {
                    'near_duration_limit': duration_minutes > (self.limits.max_duration_hours * 60 * 0.8),
                    'near_message_limit': session.message_count > (self.limits.max_messages * 0.8),
                    'near_memory_limit': session.memory_usage_mb > (self.limits.max_memory_mb * 0.8)
                }
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide session statistics"""
        with self._lock:
            status_counts = {}
            for status in SessionStatus:
                status_counts[status.value] = sum(1 for s in self.sessions.values() if s.status == status)
            
            active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
            
            if active_sessions:
                total_messages = sum(s.message_count for s in active_sessions)
                total_processing_time = sum(s.processing_time_total for s in active_sessions)
                total_memory = sum(s.memory_usage_mb for s in active_sessions)
                avg_session_duration = sum((time.time() - s.start_time) for s in active_sessions) / len(active_sessions) / 60
            else:
                total_messages = total_processing_time = total_memory = avg_session_duration = 0
            
            return {
                'total_sessions': len(self.sessions),
                'status_distribution': status_counts,
                'active_sessions': len(active_sessions),
                'concurrent_limit_utilization': len(active_sessions) / self.limits.max_concurrent_sessions,
                'total_messages_processed': total_messages,
                'total_processing_time': total_processing_time,
                'total_memory_usage_mb': total_memory,
                'avg_session_duration_minutes': avg_session_duration,
                'system_health': {
                    'overloaded': len(active_sessions) > self.limits.max_concurrent_sessions * 0.8,
                    'memory_pressure': total_memory > (self.limits.max_memory_mb * len(active_sessions) * 0.8),
                    'processing_delays': total_processing_time > (len(active_sessions) * self.limits.max_processing_time_seconds)
                }
            }
    
    def cleanup_expired_sessions(self):
        """Clean up expired and terminated sessions"""
        with self._lock:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                # Mark idle sessions as expired after extended idle time
                idle_time_hours = (current_time - session.last_activity) / 3600
                if (session.status == SessionStatus.IDLE and 
                    idle_time_hours > self.limits.idle_timeout_minutes / 60 * 2):  # 2x idle timeout
                    session.status = SessionStatus.EXPIRED
                
                # Clean up very old expired/terminated sessions
                if session.status in [SessionStatus.EXPIRED, SessionStatus.TERMINATED, SessionStatus.ERROR]:
                    session_age_hours = (current_time - session.start_time) / 3600
                    if session_age_hours > 24:  # Clean up after 24 hours
                        expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                logger.info(f"Cleaning up old session: {session_id}")
                del self.sessions[session_id]
            
            return len(expired_sessions)
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while self.running:
                try:
                    cleaned = self.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} expired sessions")
                except Exception as e:
                    logger.error(f"Error in session cleanup: {e}")
                
                time.sleep(300)  # Run every 5 minutes
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("Session cleanup thread started")
    
    def shutdown(self):
        """Shutdown session manager"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("SessionManager shutdown complete")

def test_session_manager():
    """Test session manager functionality"""
    print("Testing Session Manager")
    print("=" * 50)
    
    # Create manager with short limits for testing
    limits = SessionLimits(
        max_duration_hours=0.001,  # Very short for testing
        max_messages=3,
        max_memory_mb=10,
        idle_timeout_minutes=0.01,
        max_concurrent_sessions=2
    )
    
    manager = SessionManager(limits)
    
    # Test session creation
    session_id1, created1 = manager.create_session("user1")
    session_id2, created2 = manager.create_session("user2") 
    session_id3, created3 = manager.create_session("user3")  # Should fail
    
    print(f"Session 1 created: {created1}")
    print(f"Session 2 created: {created2}")
    print(f"Session 3 created: {created3} (should be False)")
    
    # Test validation
    valid1, reason1 = manager.validate_session(session_id1)
    print(f"Session 1 valid: {valid1}")
    
    # Test activity updates
    manager.update_session_activity(session_id1, message_processed=True, processing_time=1.0, memory_usage_mb=5.0)
    manager.update_session_activity(session_id1, message_processed=True)
    manager.update_session_activity(session_id1, message_processed=True)
    manager.update_session_activity(session_id1, message_processed=True)  # Should exceed limit
    
    # Check limits
    valid1_after, reason1_after = manager.validate_session(session_id1)
    print(f"Session 1 valid after messages: {valid1_after}, reason: {reason1_after}")
    
    # Test stats
    stats = manager.get_session_stats(session_id1)
    if stats:
        print(f"Session stats: {stats['message_count']} messages, {stats['memory_usage_mb']}MB")
    
    system_stats = manager.get_system_stats()
    print(f"System stats: {system_stats['active_sessions']} active, {system_stats['total_sessions']} total")
    
    # Cleanup
    manager.shutdown()

if __name__ == "__main__":
    test_session_manager()
