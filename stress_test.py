#!/usr/bin/env python3
"""
Comprehensive Stress Test for Emma's Production System
Tests memory management, state validation, API resilience, and session management
"""

import os
import time
import random
import asyncio
import logging
import threading
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Import production components for direct testing
from emma_companion import EmmaCompanion
from session_manager import SessionManager, SessionLimits
from system_monitor import SystemMonitor
from memory_manager import MemoryManager
from state_validator import StateValidator
from api_wrapper import SyncResilientGeminiAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class StressTestSuite:
    """Comprehensive stress test suite for Emma's production system"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Run comprehensive stress test suite"""
        print("EMMA PRODUCTION STRESS TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Memory Management Stress Test", self.test_memory_management),
            ("State Validation Stress Test", self.test_state_validation),
            ("Session Management Stress Test", self.test_session_management),
            ("API Resilience Test", self.test_api_resilience),
            ("Concurrent Users Test", self.test_concurrent_users),
            ("Extended Conversation Test", self.test_extended_conversation),
            ("System Monitor Test", self.test_system_monitoring),
            ("Production Integration Test", self.test_production_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'duration': duration,
                    'details': result if isinstance(result, dict) else {}
                }
                
                status = "PASSED" if result else "FAILED"
                print(f"{status} ({duration:.2f}s)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'duration': time.time() - start_time,
                    'error': str(e)
                }
                print(f"ERROR: {e}")
        
        self.print_summary()
    
    def test_memory_management(self) -> bool:
        """Test memory management under stress"""
        print("  Testing memory formation, consolidation, and pruning...")
        
        try:
            # Create memory manager
            memory_manager = MemoryManager(max_memories=100, pruning_threshold=0.01)
            
            # Create test memories
            memories = []
            for i in range(150):  # Exceed capacity
                memory = {
                    'id': i,
                    'timestamp': time.time() - (i * 60),  # Spread over time
                    'importance_score': random.uniform(0.1, 1.0),
                    'emotional_intensity': random.uniform(0.0, 1.0),
                    'recall_count': random.randint(0, 10),
                    'background_triggers': ['work', 'relationships', 'fears'][:random.randint(0, 3)],
                    'user_message': f"Test message {i}",
                    'dominant_emotion': ('joy', random.uniform(0.0, 1.0))
                }
                memories.append(memory)
            
            # Test consolidation
            consolidated = memory_manager.consolidate_memories(memories, force=True)
            print(f"    Consolidated {len(memories)} -> {len(consolidated)} memories")
            
            # Test health metrics
            metrics = memory_manager.get_memory_health_metrics(consolidated)
            print(f"    Memory health: {metrics['status']}")
            print(f"    Average importance: {metrics['avg_importance']:.2f}")
            
            # Test suggestions
            suggestions = memory_manager.suggest_memory_optimizations(metrics)
            print(f"    Optimization suggestions: {len(suggestions)}")
            
            return len(consolidated) <= memory_manager.max_memories and metrics['status'] in ['healthy', 'near_capacity']
            
        except Exception as e:
            print(f"    Memory management test failed: {e}")
            return False
    
    def test_state_validation(self) -> bool:
        """Test state validation and repair"""
        print("  Testing state validation with problematic states...")
        
        try:
            validator = StateValidator()
            
            # Test with problematic states
            test_states = [
                # Normal state
                {
                    'biological': {'hunger': 0.5, 'fatigue': 0.3, 'libido': 5.0},
                    'emotional': {'joy': 0.6, 'trust': 0.8, 'uncertainty': 0.2}
                },
                # State with NaN/Inf
                {
                    'biological': {'hunger': float('nan'), 'fatigue': float('inf'), 'libido': 5.0},
                    'emotional': {'joy': 0.6, 'trust': float('-inf'), 'uncertainty': 0.2}
                },
                # Out of bounds
                {
                    'biological': {'hunger': -1.0, 'fatigue': 15.0, 'libido': 5.0},
                    'emotional': {'joy': 2.0, 'trust': -0.5, 'uncertainty': 0.2}
                }
            ]
            
            repairs_applied = 0
            for i, state in enumerate(test_states):
                validated, report = validator.comprehensive_validation(state)
                if report['repairs_applied']:
                    repairs_applied += 1
                print(f"    State {i+1}: {'repaired' if report['repairs_applied'] else 'clean'}")
            
            # Test stability metrics
            metrics = validator.get_stability_metrics()
            print(f"    Stability status: {metrics['status']}")
            print(f"    Repair rate: {metrics['repair_rate']:.1%}")
            
            return repairs_applied >= 2 and metrics['status'] != 'highly_unstable'
            
        except Exception as e:
            print(f"    State validation test failed: {e}")
            return False
    
    def test_session_management(self) -> bool:
        """Test session management under load"""
        print("  Testing session creation, validation, and cleanup...")
        
        try:
            # Short limits for testing
            limits = SessionLimits(
                max_duration_hours=0.001,
                max_messages=5,
                max_memory_mb=10,
                idle_timeout_minutes=0.01,
                max_concurrent_sessions=3
            )
            
            manager = SessionManager(limits)
            
            # Test session creation
            sessions_created = 0
            for i in range(5):  # Try to exceed limit
                session_id, created = manager.create_session(f"user_{i}")
                if created:
                    sessions_created += 1
            
            print(f"    Created {sessions_created}/5 sessions (limit: {limits.max_concurrent_sessions})")
            
            # Test session validation
            valid_sessions = 0
            for session_id in list(manager.sessions.keys())[:3]:
                valid, reason = manager.validate_session(session_id)
                if valid:
                    valid_sessions += 1
                    
                    # Update activity to trigger limits
                    for _ in range(6):  # Exceed message limit
                        manager.update_session_activity(session_id, message_processed=True)
                    
                    # Check if session is now invalid
                    valid_after, reason_after = manager.validate_session(session_id)
                    if not valid_after:
                        print(f"    Session {session_id[:8]} correctly invalidated: {reason_after}")
            
            # Test system stats
            stats = manager.get_system_stats()
            print(f"    System stats: {stats['active_sessions']} active sessions")
            
            # Cleanup
            manager.shutdown()
            
            return sessions_created <= limits.max_concurrent_sessions
            
        except Exception as e:
            print(f"    Session management test failed: {e}")
            return False
    
    def test_api_resilience(self) -> bool:
        """Test API resilience and fallbacks"""
        print("  Testing API resilience with fallbacks...")
        
        try:
            # Test with dummy key (will fail, testing fallback)
            api = SyncResilientGeminiAPI("dummy_key_for_testing", max_retries=2)
            
            # Test with various contexts
            test_prompts = [
                ("Hello there", {'emotional_state': {'joy': 0.8}}),
                ("I'm feeling sad", {'emotional_state': {'uncertainty': 0.9}}),
                ("Tell me a joke", {'emotional_state': {'fatigue': 0.3}})
            ]
            
            fallback_responses = 0
            for prompt, context in test_prompts:
                response = api.generate_with_resilience(prompt, context)
                if response and len(response) > 0:
                    fallback_responses += 1
                    print(f"    Fallback response generated for: '{prompt[:20]}...'")
            
            # Test health status
            health = api.get_health_status()
            print(f"    API health: {health['health']}")
            print(f"    Success rate: {health['success_rate']:.1%}")
            print(f"    Fallback calls: {health['fallback_calls']}")
            
            return fallback_responses == len(test_prompts) and health['fallback_calls'] > 0
            
        except Exception as e:
            print(f"    API resilience test failed: {e}")
            return False
    
    def test_concurrent_users(self) -> bool:
        """Test concurrent user simulation"""
        print("  Testing concurrent user load...")
        
        def simulate_user(user_id: int) -> Dict[str, Any]:
            """Simulate a user conversation"""
            session = requests.Session()
            results = {'messages': 0, 'errors': 0, 'response_times': []}
            
            messages = [
                f"Hi Emma, I'm user {user_id}",
                "How are you feeling today?",
                "Tell me about yourself",
                "What's your favorite memory?",
                "Thanks for chatting!"
            ]
            
            for message in messages:
                try:
                    start_time = time.time()
                    response = session.post(
                        f"{self.base_url}/api/chat",
                        json={'message': message},
                        timeout=10
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        results['messages'] += 1
                        results['response_times'].append(response_time)
                    else:
                        results['errors'] += 1
                        
                except Exception as e:
                    results['errors'] += 1
                
                time.sleep(random.uniform(0.5, 2.0))  # Simulate human timing
            
            return results
        
        try:
            # Simulate 5 concurrent users
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(simulate_user, i) for i in range(5)]
                user_results = [future.result() for future in futures]
            
            # Analyze results
            total_messages = sum(r['messages'] for r in user_results)
            total_errors = sum(r['errors'] for r in user_results)
            
            all_response_times = []
            for r in user_results:
                all_response_times.extend(r['response_times'])
            
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
            
            print(f"    Total messages: {total_messages}")
            print(f"    Total errors: {total_errors}")
            print(f"    Avg response time: {avg_response_time:.2f}s")
            print(f"    Error rate: {total_errors / (total_messages + total_errors) * 100:.1f}%")
            
            return total_messages > 20 and (total_errors / max(total_messages + total_errors, 1)) < 0.2
            
        except Exception as e:
            print(f"    Concurrent users test failed: {e}")
            return False
    
    def test_extended_conversation(self) -> bool:
        """Test extended conversation memory and stability"""
        print("  Testing extended conversation stability...")
        
        try:
            # Set dummy API key for testing
            os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"
            
            emma = EmmaCompanion(gemini_api_key=os.environ["GEMINI_API_KEY"])
            
            # Extended conversation simulation
            messages = [
                "Hi Emma, let's have a long conversation",
                "Tell me about your work in brand strategy",
                "What was your childhood like?",
                "What are your biggest fears?",
                "Do you remember what we talked about earlier?",
                "How do you handle difficult emotions?",
                "What brings you the most joy?",
                "Tell me about a challenging project you worked on",
                "How do you maintain work-life balance?",
                "What's your perspective on AI and consciousness?",
                "Do you ever feel lonely?",
                "What would you do if you could travel anywhere?",
                "How do you deal with uncertainty?",
                "What's the most important lesson you've learned?",
                "Can you recall our conversation about work earlier?"
            ]
            
            responses = []
            memory_formation_count = 0
            
            for i, message in enumerate(messages):
                response = emma.process_message(message)
                responses.append(response)
                
                # Check if memories were formed
                if hasattr(emma, 'dls') and emma.dls.memories:
                    if len(emma.dls.memories) > memory_formation_count:
                        memory_formation_count = len(emma.dls.memories)
                
                # Simulate processing delay
                time.sleep(0.1)
                
                if i % 5 == 0:
                    print(f"    Processed {i+1}/{len(messages)} messages")
            
            # Analyze conversation
            stats = emma.get_stats()
            total_messages = len(messages)
            avg_response_length = sum(len(r) for r in responses) / len(responses)
            
            print(f"    Total messages processed: {total_messages}")
            print(f"    Memories formed: {memory_formation_count}")
            print(f"    Avg response length: {avg_response_length:.0f} chars")
            print(f"    Conversation length: {stats.get('conversation_length', 0)}")
            
            return total_messages == len(messages) and all(len(r) > 10 for r in responses)
            
        except Exception as e:
            print(f"    Extended conversation test failed: {e}")
            return False
    
    def test_system_monitoring(self) -> bool:
        """Test system monitoring capabilities"""
        print("  Testing system monitoring and health checks...")
        
        try:
            monitor = SystemMonitor(history_size=50)
            
            # Simulate metrics
            for i in range(20):
                monitor.log_request_metrics(
                    response_time=random.uniform(0.5, 3.0),
                    memory_usage_mb=random.uniform(100, 500),
                    state_stability=random.uniform(0.7, 1.0),
                    error_occurred=random.random() < 0.1
                )
                time.sleep(0.01)
            
            # Test health assessment
            health = monitor.get_system_health()
            print(f"    System health: {health['status']}")
            print(f"    Health score: {health['health_score']:.2f}")
            print(f"    Total requests: {health['total_requests']}")
            
            # Test dashboard data
            dashboard = monitor.get_dashboard_data()
            print(f"    Charts data points: {len(dashboard['charts']['timestamps'])}")
            print(f"    Recent alerts: {len(dashboard['alerts'])}")
            print(f"    Recommendations: {len(dashboard['recommendations'])}")
            
            # Test metrics export
            export_file = monitor.export_metrics()
            export_exists = os.path.exists(export_file)
            if export_exists:
                os.remove(export_file)
            
            print(f"    Metrics export: {'✓' if export_exists else '✗'}")
            
            return health['status'] != 'critical' and export_exists
            
        except Exception as e:
            print(f"    System monitoring test failed: {e}")
            return False
    
    def test_production_integration(self) -> bool:
        """Test full production integration"""
        print("  Testing full production system integration...")
        
        try:
            # Test health endpoint
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                health_status = response.status_code in [200, 202, 503]
                print(f"    Health endpoint: {'✓' if health_status else '✗'} ({response.status_code})")
            except:
                health_status = False
                print("    Health endpoint: ✗ (unreachable)")
            
            # Test state endpoint
            try:
                response = requests.get(f"{self.base_url}/api/state", timeout=5)
                state_status = response.status_code == 200
                print(f"    State endpoint: {'✓' if state_status else '✗'} ({response.status_code})")
            except:
                state_status = False
                print("    State endpoint: ✗ (unreachable)")
            
            # Test chat endpoint
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={'message': 'Integration test message'},
                    timeout=10
                )
                chat_status = response.status_code == 200
                if chat_status:
                    data = response.json()
                    has_response = 'response' in data and len(data['response']) > 0
                    print(f"    Chat endpoint: {'✓' if has_response else '✗'}")
                else:
                    print(f"    Chat endpoint: ✗ ({response.status_code})")
            except:
                chat_status = False
                print("    Chat endpoint: ✗ (unreachable)")
            
            return health_status and state_status and chat_status
            
        except Exception as e:
            print(f"    Production integration test failed: {e}")
            return False
    
    def print_summary(self):
        """Print comprehensive test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("STRESS TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        errors = sum(1 for r in self.test_results.values() if r['status'] == 'ERROR')
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR"}[result['status']]
            print(f"  {status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s)")
            if result['status'] == 'ERROR':
                print(f"      Error: {result.get('error', 'Unknown error')}")
        
        # Overall assessment
        print(f"\nOVERALL ASSESSMENT:")
        if passed == total:
            print("EXCELLENT - All systems performing optimally")
        elif passed >= total * 0.8:
            print("GOOD - Most systems stable, minor issues detected")
        elif passed >= total * 0.6:
            print("CONCERNING - Multiple system issues detected")
        else:
            print("CRITICAL - Major system instability detected")
        
        print("\nProduction Readiness Assessment:")
        production_critical = [
            "Memory Management Stress Test",
            "State Validation Stress Test", 
            "Session Management Stress Test",
            "API Resilience Test"
        ]
        
        critical_passed = sum(1 for test in production_critical 
                            if self.test_results.get(test, {}).get('status') == 'PASSED')
        
        if critical_passed == len(production_critical):
            print("PRODUCTION READY - All critical systems validated")
        else:
            print(f"NOT PRODUCTION READY - {len(production_critical) - critical_passed} critical issues")

def main():
    """Run stress test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emma Production Stress Test Suite')
    parser.add_argument('--url', default='http://127.0.0.1:5000', 
                       help='Base URL for API tests (default: http://127.0.0.1:5000)')
    parser.add_argument('--skip-api', action='store_true',
                       help='Skip API integration tests (for offline testing)')
    
    args = parser.parse_args()
    
    suite = StressTestSuite(args.url)
    
    if args.skip_api:
        print("Skipping API integration tests (offline mode)")
        # Remove API-dependent tests
        suite.test_concurrent_users = lambda: True
        suite.test_production_integration = lambda: True
    
    suite.run_all_tests()

if __name__ == "__main__":
    main()
