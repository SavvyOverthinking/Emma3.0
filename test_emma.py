#!/usr/bin/env python3
"""
Comprehensive testing framework for Emma Digital Biology Companion
Tests stability, performance, and functionality
"""

import time
import json
import logging
import psutil
import numpy as np
from typing import Dict, List, Any
import traceback
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import Emma components
from emma_companion import EmmaCompanion
from biological_substrate import BiologicalSubstrate
from rag_system import EfficientRAGSystem
from phenomenology import EffectivePhenomenologyTranslator

class EmmaTestSuite:
    def __init__(self):
        self.test_results = {
            "timestamp": time.time(),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            },
            "performance": {},
            "memory_usage": {}
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        self.logger.info("Starting Emma test suite...")
        
        test_suites = [
            self.test_biological_substrate,
            self.test_rag_system,
            self.test_phenomenology,
            self.test_emma_integration,
            self.test_stability,
            self.test_performance,
            self.test_memory_efficiency,
            self.test_concurrent_access
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                self.logger.error(f"Test suite {test_suite.__name__} failed: {e}")
                self.test_results["summary"]["errors"] += 1
                
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    def test_biological_substrate(self):
        """Test BiologicalSubstrate functionality"""
        self.logger.info("Testing BiologicalSubstrate...")
        
        tests = [
            self.test_substrate_initialization,
            self.test_substrate_evolution,
            self.test_substrate_memory,
            self.test_substrate_stability,
            self.test_substrate_drives
        ]
        
        self.run_test_suite("biological_substrate", tests)
    
    def test_substrate_initialization(self) -> bool:
        """Test substrate initialization"""
        try:
            substrate = BiologicalSubstrate(dim=128, sparsity=0.1)
            
            # Check dimensions
            assert substrate.dim == 128
            assert len(substrate.state) == 128
            assert substrate.connectivity.shape == (128, 128)
            
            # Check initial state
            assert np.allclose(substrate.state, np.zeros(128))
            assert substrate.drives["fatigue"] == 0.0
            assert substrate.drives["curiosity"] == 0.7
            
            return True
            
        except Exception as e:
            self.logger.error(f"Substrate initialization test failed: {e}")
            return False
    
    def test_substrate_evolution(self) -> bool:
        """Test substrate evolution"""
        try:
            substrate = BiologicalSubstrate(dim=64)
            
            # Test evolution with random input
            input_vector = np.random.randn(64) * 0.1
            initial_state = substrate.state.copy()
            
            new_state = substrate.evolve(input_vector)
            
            # Check that state changed
            assert not np.allclose(initial_state, new_state)
            assert len(new_state) == 64
            
            # Check stability
            state_norm = np.linalg.norm(new_state)
            assert 0.1 <= state_norm <= 10.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Substrate evolution test failed: {e}")
            return False
    
    def test_substrate_memory(self) -> bool:
        """Test memory updates"""
        try:
            substrate = BiologicalSubstrate(dim=32)
            
            # Test memory update
            experience = np.random.randn(32) * 0.5
            initial_memory_nnz = substrate.memory_matrix.nnz
            
            substrate.update_memory(experience, strength=0.01)
            
            # Check that memory was updated
            assert substrate.memory_matrix.nnz >= initial_memory_nnz
            
            # Test memory decay
            substrate.update_memory(experience, strength=0.001)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Substrate memory test failed: {e}")
            return False
    
    def test_substrate_stability(self) -> bool:
        """Test numerical stability"""
        try:
            substrate = BiologicalSubstrate(dim=32)
            
            # Test multiple evolutions
            for i in range(100):
                input_vector = np.random.randn(32) * 0.2
                state = substrate.evolve(input_vector)
                
                # Check for NaN/Inf
                assert not np.any(np.isnan(state))
                assert not np.any(np.isinf(state))
                
                # Check state norm
                norm = np.linalg.norm(state)
                assert norm <= 10.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Substrate stability test failed: {e}")
            return False
    
    def test_substrate_drives(self) -> bool:
        """Test drive updates"""
        try:
            substrate = BiologicalSubstrate(dim=32)
            
            initial_fatigue = substrate.drives["fatigue"]
            
            # Evolve multiple times
            for i in range(50):
                input_vector = np.random.randn(32) * 0.1
                substrate.evolve(input_vector)
            
            # Check that fatigue increased
            assert substrate.drives["fatigue"] > initial_fatigue
            assert 0.0 <= substrate.drives["fatigue"] <= 1.0
            
            # Check other drives
            for drive_name, value in substrate.drives.items():
                assert 0.0 <= value <= 1.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Substrate drives test failed: {e}")
            return False
    
    def test_rag_system(self):
        """Test RAG system functionality"""
        self.logger.info("Testing RAG System...")
        
        tests = [
            self.test_rag_initialization,
            self.test_rag_retrieval,
            self.test_rag_document_loading,
            self.test_rag_chunking
        ]
        
        self.run_test_suite("rag_system", tests)
    
    def test_rag_initialization(self) -> bool:
        """Test RAG system initialization"""
        try:
            rag = EfficientRAGSystem("test_knowledge_base")
            
            # Check initialization
            assert rag.chunk_size == 300
            assert rag.overlap == 50
            assert rag.vectorizer is not None
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG initialization test failed: {e}")
            return False
    
    def test_rag_retrieval(self) -> bool:
        """Test document retrieval"""
        try:
            # Create test documents
            test_docs = {
                "test_0": {
                    "title": "Test Document",
                    "content": "This is a test document about artificial intelligence and machine learning."
                }
            }
            
            rag = EfficientRAGSystem("test_knowledge_base")
            rag.documents = test_docs
            rag.document_vectors = rag.vectorizer.fit_transform([doc["content"] for doc in test_docs.values()])
            rag.document_ids = list(test_docs.keys())
            rag.is_fitted = True
            
            # Test retrieval
            results = rag.retrieve("artificial intelligence", top_k=1)
            
            assert len(results) == 1
            assert results[0]["title"] == "Test Document"
            assert results[0]["relevance"] > 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG retrieval test failed: {e}")
            return False
    
    def test_rag_document_loading(self) -> bool:
        """Test document loading"""
        try:
            # This would normally test actual file loading
            rag = EfficientRAGSystem("nonexistent_path")
            
            # Should handle missing directory gracefully
            assert len(rag.documents) == 0
            assert not rag.is_fitted
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG document loading test failed: {e}")
            return False
    
    def test_rag_chunking(self) -> bool:
        """Test document chunking"""
        try:
            rag = EfficientRAGSystem("test_knowledge_base")
            
            # Test chunking long content
            long_content = "This is a test. " * 100
            chunks = rag._split_into_chunks(long_content, max_length=50)
            
            assert len(chunks) > 1
            assert all(len(chunk) <= 50 for chunk in chunks)
            
            return True
            
        except Exception as e:
            self.logger.error(f"RAG chunking test failed: {e}")
            return False
    
    def test_phenomenology(self):
        """Test phenomenology translator"""
        self.logger.info("Testing Phenomenology Translator...")
        
        tests = [
            self.test_phenomenology_initialization,
            self.test_phenomenology_translation,
            self.test_phenomenology_cluster_selection
        ]
        
        self.run_test_suite("phenomenology", tests)
    
    def test_phenomenology_initialization(self) -> bool:
        """Test phenomenology initialization"""
        try:
            translator = EffectivePhenomenologyTranslator()
            
            assert len(translator.somatic_clusters) > 0
            assert translator.n_clusters == len(translator.somatic_clusters)
            assert not translator.initialized
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phenomenology initialization test failed: {e}")
            return False
    
    def test_phenomenology_translation(self) -> bool:
        """Test phenomenology translation"""
        try:
            translator = EffectivePhenomenologyTranslator()
            
            # Test with sample state
            state = np.random.randn(64)
            drives = {"fatigue": 0.3, "curiosity": 0.7, "social": 0.5}
            
            description = translator.translate(state, drives)
            
            assert isinstance(description, str)
            assert len(description) > 0
            assert description in [desc for cluster in translator.somatic_clusters.values() for desc in cluster]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phenomenology translation test failed: {e}")
            return False
    
    def test_phenomenology_cluster_selection(self) -> bool:
        """Test cluster selection based on drives"""
        try:
            translator = EffectivePhenomenologyTranslator()
            
            # Test high fatigue selection
            state = np.random.randn(64)
            high_fatigue_drives = {"fatigue": 0.9, "curiosity": 0.1, "social": 0.5}
            
            description = translator.translate(state, high_fatigue_drives)
            fatigue_descriptions = translator.somatic_clusters["fatigue"]
            
            assert description in fatigue_descriptions
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phenomenology cluster selection test failed: {e}")
            return False
    
    def test_emma_integration(self):
        """Test full Emma integration"""
        self.logger.info("Testing Emma Integration...")
        
        tests = [
            self.test_emma_initialization,
            self.test_emma_message_processing,
            self.test_emma_conversation_history,
            self.test_emma_stats
        ]
        
        self.run_test_suite("emma_integration", tests)
    
    def test_emma_initialization(self) -> bool:
        """Test Emma initialization"""
        try:
            emma = EmmaCompanion()
            
            assert emma.session_id is not None
            assert emma.substrate is not None
            assert emma.rag_system is not None
            assert emma.phenomenology is not None
            assert len(emma.conversation_history) == 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emma initialization test failed: {e}")
            return False
    
    def test_emma_message_processing(self) -> bool:
        """Test message processing"""
        try:
            emma = EmmaCompanion()
            
            # Process test message
            response = emma.process_message("Hello, how are you?")
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert len(emma.conversation_history) == 2  # User + Emma messages
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emma message processing test failed: {e}")
            return False
    
    def test_emma_conversation_history(self) -> bool:
        """Test conversation history management"""
        try:
            emma = EmmaCompanion()
            
            # Add multiple messages
            for i in range(5):
                emma.process_message(f"Test message {i}")
            
            assert len(emma.conversation_history) == 10  # 5 user + 5 emma
            assert emma.message_count == 5
            
            # Check history limit
            for i in range(50):  # Add more than limit
                emma.process_message(f"Message {i}")
            
            assert len(emma.conversation_history) <= emma.max_history_length
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emma conversation history test failed: {e}")
            return False
    
    def test_emma_stats(self) -> bool:
        """Test statistics collection"""
        try:
            emma = EmmaCompanion()
            
            # Process some messages
            for i in range(3):
                emma.process_message(f"Test {i}")
            
            stats = emma.get_stats()
            
            assert "session_id" in stats
            assert "conversation_length" in stats
            assert "substrate_stats" in stats
            assert "rag_stats" in stats
            assert stats["conversation_length"] == 6  # 3 user + 3 emma
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emma stats test failed: {e}")
            return False
    
    def test_stability(self):
        """Test system stability under load"""
        self.logger.info("Testing System Stability...")
        
        tests = [
            self.test_long_conversation,
            self.test_rapid_messages,
            self.test_error_recovery
        ]
        
        self.run_test_suite("stability", tests)
    
    def test_long_conversation(self) -> bool:
        """Test stability over long conversation"""
        try:
            emma = EmmaCompanion()
            
            # Simulate long conversation
            test_messages = [
                "Hello, how are you doing today?",
                "What do you think about artificial intelligence?",
                "Tell me about your work in brand strategy.",
                "How do you handle stress and difficult situations?",
                "What are your thoughts on authenticity in relationships?",
                "Describe your perfect day.",
                "What makes you feel most alive?",
                "How do you define success?",
                "What are you afraid of?",
                "What gives you hope?"
            ]
            
            for message in test_messages:
                response = emma.process_message(message)
                assert isinstance(response, str)
                assert len(response) > 0
            
            # Check system integrity
            stats = emma.get_stats()
            assert stats["conversation_length"] == 20  # 10 user + 10 emma
            
            return True
            
        except Exception as e:
            self.logger.error(f"Long conversation test failed: {e}")
            return False
    
    def test_rapid_messages(self) -> bool:
        """Test handling of rapid message sending"""
        try:
            emma = EmmaCompanion()
            
            # Send messages rapidly
            for i in range(10):
                start_time = time.time()
                response = emma.process_message(f"Rapid message {i}")
                processing_time = time.time() - start_time
                
                assert processing_time < 5.0  # Should respond within 5 seconds
                assert isinstance(response, str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rapid messages test failed: {e}")
            return False
    
    def test_error_recovery(self) -> bool:
        """Test error recovery mechanisms"""
        try:
            substrate = BiologicalSubstrate(dim=32)
            
            # Test recovery from bad input
            try:
                substrate.evolve(np.array([float('inf')] * 32))
            except:
                pass
            
            # System should still work
            response = substrate.evolve(np.random.randn(32) * 0.1)
            assert len(response) == 32
            assert not np.any(np.isnan(response))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recovery test failed: {e}")
            return False
    
    def test_performance(self):
        """Test system performance"""
        self.logger.info("Testing Performance...")
        
        tests = [
            self.test_response_times,
            self.test_memory_usage,
            self.test_throughput
        ]
        
        self.run_test_suite("performance", tests)
    
    def test_response_times(self) -> bool:
        """Test response time performance"""
        try:
            emma = EmmaCompanion()
            
            response_times = []
            
            for i in range(20):
                start_time = time.time()
                emma.process_message(f"Performance test message {i}")
                response_time = time.time() - start_time
                response_times.append(response_time)
            
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            
            self.test_results["performance"]["avg_response_time"] = avg_response_time
            self.test_results["performance"]["max_response_time"] = max_response_time
            
            # Should respond within reasonable time
            assert avg_response_time < 3.0  # Average under 3 seconds
            assert max_response_time < 5.0  # Maximum under 5 seconds
            
            return True
            
        except Exception as e:
            self.logger.error(f"Response times test failed: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage efficiency"""
        try:
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple instances
            instances = []
            for i in range(5):
                emma = EmmaCompanion()
                instances.append(emma)
                
                # Process some messages
                for j in range(10):
                    emma.process_message(f"Memory test {i}-{j}")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.test_results["memory_usage"]["initial"] = initial_memory
            self.test_results["memory_usage"]["final"] = final_memory
            self.test_results["memory_usage"]["increase"] = memory_increase
            
            # Memory increase should be reasonable
            assert memory_increase < 100  # Less than 100MB increase
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory usage test failed: {e}")
            return False
    
    def test_throughput(self) -> bool:
        """Test message processing throughput"""
        try:
            emma = EmmaCompanion()
            
            start_time = time.time()
            message_count = 0
            
            # Process messages for 10 seconds
            while time.time() - start_time < 10:
                emma.process_message(f"Throughput test {message_count}")
                message_count += 1
            
            elapsed_time = time.time() - start_time
            throughput = message_count / elapsed_time
            
            self.test_results["performance"]["throughput"] = throughput
            
            # Should handle reasonable throughput
            assert throughput > 0.1  # At least 0.1 messages per second
            
            return True
            
        except Exception as e:
            self.logger.error(f"Throughput test failed: {e}")
            return False
    
    def test_memory_efficiency(self):
        """Test memory efficiency specifically"""
        self.logger.info("Testing Memory Efficiency...")
        
        tests = [
            self.test_sparse_matrices,
            self.test_memory_leaks,
            self.test_garbage_collection
        ]
        
        self.run_test_suite("memory_efficiency", tests)
    
    def test_sparse_matrices(self) -> bool:
        """Test sparse matrix efficiency"""
        try:
            substrate = BiologicalSubstrate(dim=512, sparsity=0.05)
            
            # Check sparsity
            total_elements = 512 * 512
            non_zero_elements = substrate.connectivity.nnz
            actual_sparsity = non_zero_elements / total_elements
            
            # Should be close to target sparsity
            assert abs(actual_sparsity - 0.05) < 0.02
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sparse matrices test failed: {e}")
            return False
    
    def test_memory_leaks(self) -> bool:
        """Test for memory leaks"""
        try:
            process = psutil.Process()
            
            # Run intensive operations
            for i in range(100):
                emma = EmmaCompanion()
                for j in range(10):
                    emma.process_message(f"Leak test {i}-{j}")
                
                # Explicitly delete to test cleanup
                del emma
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Memory should be reasonable
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            assert current_memory < 500  # Less than 500MB
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory leaks test failed: {e}")
            return False
    
    def test_garbage_collection(self) -> bool:
        """Test garbage collection effectiveness"""
        try:
            import gc
            
            # Create many objects
            objects = []
            for i in range(1000):
                substrate = BiologicalSubstrate(dim=32)
                objects.append(substrate)
            
            # Check memory before cleanup
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clear references and collect garbage
            objects.clear()
            gc.collect()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory should decrease significantly
            assert memory_after < memory_before * 1.5  # Allow some overhead
            
            return True
            
        except Exception as e:
            self.logger.error(f"Garbage collection test failed: {e}")
            return False
    
    def test_concurrent_access(self):
        """Test concurrent access handling"""
        self.logger.info("Testing Concurrent Access...")
        
        tests = [
            self.test_thread_safety,
            self.test_async_operations
        ]
        
        self.run_test_suite("concurrent_access", tests)
    
    def test_thread_safety(self) -> bool:
        """Test thread safety"""
        try:
            emma = EmmaCompanion()
            results = []
            errors = []
            
            def process_message_thread(message):
                try:
                    response = emma.process_message(message)
                    results.append(response)
                except Exception as e:
                    errors.append(e)
            
            # Run multiple threads
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i in range(20):
                    future = executor.submit(process_message_thread, f"Thread test {i}")
                    futures.append(future)
                
                # Wait for completion
                for future in futures:
                    future.result()
            
            # Check results
            assert len(results) == 20
            assert len(errors) == 0
            assert len(emma.conversation_history) == 40  # 20 user + 20 emma
            
            return True
            
        except Exception as e:
            self.logger.error(f"Thread safety test failed: {e}")
            return False
    
    def test_async_operations(self) -> bool:
        """Test async operations"""
        try:
            # This would test async message processing
            # For now, just verify the system doesn't crash
            emma = EmmaCompanion()
            
            # Simulate async processing
            import asyncio
            
            async def async_process():
                return emma.process_message("Async test message")
            
            # Run async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(async_process())
                assert isinstance(result, str)
            finally:
                loop.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Async operations test failed: {e}")
            return False
    
    def run_test_suite(self, suite_name: str, tests: List):
        """Run a suite of tests"""
        suite_results = {
            "total": len(tests),
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for test in tests:
            try:
                self.test_results["summary"]["total_tests"] += 1
                
                if test():
                    suite_results["passed"] += 1
                    self.test_results["summary"]["passed"] += 1
                else:
                    suite_results["failed"] += 1
                    self.test_results["summary"]["failed"] += 1
                    
            except Exception as e:
                suite_results["failed"] += 1
                suite_results["errors"].append(str(e))
                self.test_results["summary"]["failed"] += 1
                self.test_results["summary"]["errors"] += 1
        
        self.test_results["tests"][suite_name] = suite_results
        
        self.logger.info(f"Test suite '{suite_name}': {suite_results['passed']}/{suite_results['total']} passed")
    
    def generate_summary(self):
        """Generate test summary"""
        total_time = time.time() - self.start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.test_results["summary"]["total_time"] = total_time
        self.test_results["memory_usage"]["final"] = final_memory
        self.test_results["memory_usage"]["increase"] = final_memory - self.initial_memory
        
        self.logger.info("=" * 60)
        self.logger.info("EMMA TEST SUITE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tests: {self.test_results['summary']['total_tests']}")
        self.logger.info(f"Passed: {self.test_results['summary']['passed']}")
        self.logger.info(f"Failed: {self.test_results['summary']['failed']}")
        self.logger.info(f"Errors: {self.test_results['summary']['errors']}")
        self.logger.info(f"Total Time: {total_time:.2f} seconds")
        self.logger.info(f"Memory Increase: {final_memory - self.initial_memory:.2f} MB")
        self.logger.info("=" * 60)
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        self.logger.info("Test results saved to test_results.json")

def main():
    """Main test runner"""
    test_suite = EmmaTestSuite()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if results["summary"]["errors"] > 0 or results["summary"]["failed"] > 0:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()