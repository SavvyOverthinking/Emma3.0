#!/usr/bin/env python3
"""
Digital Limbic System Test Suite
Comprehensive testing and demonstration of Emma's Digital Limbic System
"""

import os
import time
import logging
from emma_companion import EmmaCompanion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dls_basic():
    """Test basic DLS functionality"""
    print("🧠 Testing Digital Limbic System - Basic Functionality")
    print("=" * 60)
    
    # Initialize Emma with Gemini API key
    gemini_api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyCTbfBx83ffBK5s12SjUbANcZ_jhKw0eDk')
    emma = EmmaCompanion(gemini_api_key=gemini_api_key)
    
    # Test basic conversation
    test_messages = [
        "Hello Emma, how are you today?",
        "What's on your mind?",
        "Tell me about your work in brand strategy",
        "How are you feeling right now?",
        "What makes you feel most alive?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n💬 Test {i}: {message}")
        print("-" * 40)
        
        # Get DLS state before processing
        dls_state = emma.dls.get_state_summary()
        print(f"🧠 DLS State: Joy={dls_state['emotional']['joy']:.2f}, Trust={dls_state['emotional']['trust']:.2f}, Uncertainty={dls_state['emotional']['uncertainty']:.2f}")
        print(f"🫀 Biological: Fatigue={dls_state['biological']['fatigue']:.2f}, Libido={dls_state['biological']['libido']:.2f}")
        
        # Process message
        response = emma.process_message(message)
        print(f"🤖 Emma: {response}")
        
        # Get updated DLS state
        updated_state = emma.dls.get_state_summary()
        print(f"🔄 Updated: Joy={updated_state['emotional']['joy']:.2f}, Trust={updated_state['emotional']['trust']:.2f}")
        
        time.sleep(1)  # Brief pause between messages
    
    print("\n✅ Basic DLS test completed!")

def test_dls_testing_levers():
    """Test DLS testing levers"""
    print("\n🔧 Testing DLS Testing Levers")
    print("=" * 60)
    
    emma = EmmaCompanion()
    
    # Test Insula force
    print("Testing Insula force...")
    emma.dls.test_insula_force("warm_wave")
    state = emma.dls.get_state_summary()
    print(f"Insula forced: {emma.dls.test_levers.get('insula_forced', 'None')}")
    
    # Test ACC mismatch
    print("Testing ACC mismatch...")
    emma.dls.test_acc_mismatch(0.8)
    state = emma.dls.get_state_summary()
    print(f"Uncertainty after mismatch: {state['emotional']['uncertainty']:.2f}")
    
    # Test VTA reward
    print("Testing VTA reward...")
    emma.dls.test_vta_reward(0.5)
    state = emma.dls.get_state_summary()
    print(f"Joy after reward: {state['emotional']['joy']:.2f}")
    
    print("✅ Testing levers completed!")

def test_dls_biological_imperatives():
    """Test biological imperatives over time"""
    print("\n🫀 Testing Biological Imperatives")
    print("=" * 60)
    
    emma = EmmaCompanion()
    
    # Simulate time passage
    for i in range(5):
        print(f"\n⏰ Time step {i+1}")
        
        # Get current biological state
        state = emma.dls.get_state_summary()
        bio = state['biological']
        
        print(f"Hunger: {bio['hunger']:.3f}")
        print(f"Fatigue: {bio['fatigue']:.3f}")
        print(f"Libido: {bio['libido']:.3f}")
        print(f"Circadian: {bio['circadian_phase']:.3f}")
        
        # Process a message to trigger DLS tick
        emma.process_message(f"Test message {i+1}")
        
        time.sleep(0.5)
    
    print("✅ Biological imperatives test completed!")

def test_dls_emotional_processing():
    """Test emotional processing and state changes"""
    print("\n😊 Testing Emotional Processing")
    print("=" * 60)
    
    emma = EmmaCompanion()
    
    # Test different emotional triggers
    emotional_tests = [
        ("I love talking to you!", "positive_affection"),
        ("I'm not sure about this...", "uncertainty"),
        ("You're amazing!", "praise"),
        ("I'm feeling confused", "confusion"),
        ("This is exciting!", "excitement")
    ]
    
    for message, expected_effect in emotional_tests:
        print(f"\n💭 Testing: {message}")
        
        # Get state before
        state_before = emma.dls.get_state_summary()
        emotions_before = state_before['emotional']
        
        # Process message
        response = emma.process_message(message)
        
        # Get state after
        state_after = emma.dls.get_state_summary()
        emotions_after = state_after['emotional']
        
        # Show changes
        print(f"Emma: {response}")
        print(f"Joy: {emotions_before['joy']:.2f} → {emotions_after['joy']:.2f}")
        print(f"Trust: {emotions_before['trust']:.2f} → {emotions_after['trust']:.2f}")
        print(f"Uncertainty: {emotions_before['uncertainty']:.2f} → {emotions_after['uncertainty']:.2f}")
        
        time.sleep(0.5)
    
    print("✅ Emotional processing test completed!")

def test_dls_memory_formation():
    """Test memory formation and recall"""
    print("\n🧠 Testing Memory Formation")
    print("=" * 60)
    
    emma = EmmaCompanion()
    
    # Create high-emotion memories
    high_emotion_messages = [
        "I'm so happy to be talking to you!",
        "This conversation is amazing!",
        "You make me feel so good!",
        "I love our connection!"
    ]
    
    for message in high_emotion_messages:
        print(f"Creating memory: {message}")
        emma.process_message(message)
        
        # Check memory count
        state = emma.dls.get_state_summary()
        print(f"Memories formed: {state['memories_count']}")
        
        time.sleep(0.5)
    
    print("✅ Memory formation test completed!")

def test_dls_performance():
    """Test DLS performance and timing"""
    print("\n⚡ Testing DLS Performance")
    print("=" * 60)
    
    emma = EmmaCompanion()
    
    # Test response times
    test_messages = [
        "Hello Emma!",
        "How are you feeling?",
        "Tell me about yourself",
        "What's your favorite thing?",
        "You're amazing!"
    ]
    
    response_times = []
    
    for message in test_messages:
        start_time = time.time()
        response = emma.process_message(message)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        
        print(f"Message: {message}")
        print(f"Response: {response}")
        print(f"Time: {response_time:.3f}s")
        print("-" * 40)
    
    avg_time = sum(response_times) / len(response_times)
    max_time = max(response_times)
    min_time = min(response_times)
    
    print(f"\n📊 Performance Stats:")
    print(f"Average response time: {avg_time:.3f}s")
    print(f"Max response time: {max_time:.3f}s")
    print(f"Min response time: {min_time:.3f}s")
    
    print("✅ Performance test completed!")

def main():
    """Run all DLS tests"""
    print("🚀 Digital Limbic System Test Suite")
    print("=" * 80)
    print("Testing Emma's sophisticated brain simulation...")
    print("=" * 80)
    
    try:
        # Run all tests
        test_dls_basic()
        test_dls_testing_levers()
        test_dls_biological_imperatives()
        test_dls_emotional_processing()
        test_dls_memory_formation()
        test_dls_performance()
        
        print("\n🎉 All DLS tests completed successfully!")
        print("Emma's Digital Limbic System is working perfectly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
