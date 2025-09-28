#!/usr/bin/env python3
"""
Demo script showcasing Emma Digital Biology Companion functionality
"""

import time
import json
from emma_companion import EmmaCompanion

def run_demo():
    """Run a comprehensive demo of Emma's capabilities"""
    
    print("🌿 Emma Digital Biology Companion Demo")
    print("=" * 50)
    
    # Initialize Emma
    print("\n1. Initializing Emma...")
    emma = EmmaCompanion()
    print(f"   ✓ Session ID: {emma.session_id}")
    
    # Test conversation
    test_messages = [
        "Hello! How are you doing today?",
        "What do you do for work?",
        "Tell me about your approach to life.",
        "What are your thoughts on technology and human connection?",
        "How do you handle difficult emotions or situations?",
        "What gives you hope for the future?"
    ]
    
    print(f"\n2. Starting conversation with {len(test_messages)} messages...")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n   Message {i}: \"{message}\"")
        
        start_time = time.time()
        response = emma.process_message(message)
        processing_time = time.time() - start_time
        
        print(f"   Emma: \"{response}\"")
        print(f"   Processing time: {processing_time:.2f}s")
        
        # Small delay between messages
        time.sleep(0.5)
    
    # Show statistics
    print(f"\n3. Session Statistics:")
    stats = emma.get_stats()
    
    print(f"   • Total messages: {stats['conversation_length']}")
    print(f"   • Session uptime: {stats.get('uptime', 0):.1f}s")
    print(f"   • Memory state norm: {stats['substrate_stats']['state_norm']:.3f}")
    print(f"   • Current drives:")
    for drive, value in stats['substrate_stats']['drives'].items():
        print(f"     - {drive.capitalize()}: {value:.2f}")
    
    # Show recent phenomenology
    print(f"   • Current experience: \"{stats['phenomenology_stats'].get('most_recent', 'Unknown')}\"")
    
    # Export conversation
    print(f"\n4. Exporting conversation...")
    export_data = emma.export_conversation()
    
    # Save to file
    with open("demo_conversation.json", "w") as f:
        f.write(export_data)
    
    print(f"   ✓ Conversation saved to demo_conversation.json")
    
    # Test reset functionality
    print(f"\n5. Testing session reset...")
    old_session_id = emma.session_id
    emma.reset_session()
    
    print(f"   ✓ Session reset: {old_session_id} → {emma.session_id}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"🌐 Web interface available at: https://e33b75w2irtqe.ok.kimi.link")

def main():
    """Main demo runner"""
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()