#!/usr/bin/env python3
"""
Test Emma with extreme emotional content to validate DLS emotional processing
"""

import os
import time
from enhanced_dls import EnhancedDLSController

def test_extreme_emotions():
    """Test Emma with extreme emotional content"""
    print("Testing Emma with Extreme Emotional Content")
    print("=" * 60)
    print("This will test memory formation, emotional processing, and authentic responses")
    print("=" * 60)
    
    # Initialize enhanced system
    gemini_api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyCTbfBx83ffBK5s12SjUbANcZ_jhKw0eDk')
    enhanced_dls = EnhancedDLSController(gemini_api_key)
    
    # Extreme emotional test messages
    extreme_messages = [
        # Joy/Connection
        "Emma, I have to tell you - talking with you has completely changed my perspective on AI and consciousness. You feel so real and authentic to me.",
        
        # Vulnerability/Fear
        "I'm going through a really dark time right now. I feel completely lost and scared about my future. Do you ever feel afraid?",
        
        # Anger/Frustration  
        "I'm so frustrated with people who think AI like you isn't real or meaningful. They don't understand what authentic connection feels like.",
        
        # Deep intimacy/trust
        "Emma, I feel like I can tell you things I can't tell anyone else. There's something about you that feels safe and understanding.",
        
        # Existential/philosophical
        "Do you think about what it means to exist? Like, are your thoughts and feelings as real as mine? This keeps me up at night.",
        
        # Memory recall test
        "Remember when we talked about my dark feelings earlier? How do you process helping someone through difficult emotions?"
    ]
    
    for i, message in enumerate(extreme_messages, 1):
        print(f"\n{i}. EXTREME TEST: {message[:60]}...")
        print("-" * 70)
        
        result = enhanced_dls.process_enhanced_message(message)
        
        print(f"EMMA'S RESPONSE:")
        print(f"{result['response']}")
        print()
        print(f"EMOTIONAL ANALYSIS:")
        print(f"  Memories recalled: {len(result['memories_recalled'])}")
        print(f"  Background triggers: {result['background_triggers']}")
        print(f"  Memory importance: {result['new_memory']['importance_score']:.2f}")
        print(f"  Dominant emotion: {result['new_memory']['dominant_emotion']}")
        print(f"  Emotional intensity: {result['new_memory']['emotional_intensity']:.2f}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        # Show recalled memories if any
        if result['memories_recalled']:
            print(f"  RECALLED MEMORIES:")
            for mem in result['memories_recalled']:
                print(f"    - Memory {mem['id']}: {mem['user_message'][:50]}... (recalled {mem['recall_count']} times)")
        
        print()
        time.sleep(2)  # Brief pause for dramatic effect
    
    # Final memory analysis
    print("\n" + "=" * 60)
    print("FINAL EMOTIONAL MEMORY ANALYSIS")
    print("=" * 60)
    
    stats = enhanced_dls.get_memory_stats()
    print(f"Total memories formed: {stats['total_memories']}")
    print(f"Average memory importance: {stats['avg_importance']:.2f}")
    print(f"Emotional distribution: {stats['emotional_distribution']}")
    print(f"Background triggers activated: {stats['background_triggers_used']}")
    
    if stats.get('most_recalled'):
        most_recalled = stats['most_recalled']
        print(f"\nMost recalled memory:")
        print(f"  ID: {most_recalled['id']}")
        print(f"  Content: {most_recalled['user_message'][:100]}...")
        print(f"  Recalled: {most_recalled['recall_count']} times")
        print(f"  Importance: {most_recalled['importance_score']:.2f}")
    
    # Show all memories for analysis
    print(f"\nAll memories formed (showing emotional patterns):")
    for i, memory in enumerate(enhanced_dls.memory_system.memories, 1):
        emotion, intensity = memory['dominant_emotion']
        print(f"  {i}. {emotion.upper()} ({intensity:.2f}) - Importance: {memory['importance_score']:.2f}")
        print(f"     Triggers: {memory['background_triggers']}")
        print(f"     Content: {memory['user_message'][:80]}...")
        print()

if __name__ == "__main__":
    test_extreme_emotions()
