#!/usr/bin/env python3
"""
Enhanced Digital Limbic System with Memory Integration and Background Knowledge
"""

import os
import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dls_core import DigitalLimbicSystem, DLSPayload
from emma_companion import EmmaCompanion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMemorySystem:
    """Enhanced memory system with emotional tagging and background integration"""
    
    def __init__(self):
        self.memories = []
        self.background_knowledge = self._load_background_knowledge()
        self.memory_id_counter = 0
        self.emotional_patterns = {}
        
    def _load_background_knowledge(self) -> Dict[str, str]:
        """Load Emma's background knowledge from files"""
        knowledge = {}
        knowledge_files = {
            'core_identity': 'knowledge_base/emma_core_identity.md',
            'childhood': 'knowledge_base/emma_childhood_memories.md', 
            'fears': 'knowledge_base/emma_fears_phobias.md',
            'appearance': 'knowledge_base/emma_appearance.md',
            'intimacy': 'knowledge_base/emma_intimacy.md'
        }
        
        for key, filepath in knowledge_files.items():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    knowledge[key] = f.read()
            except FileNotFoundError:
                logger.warning(f"Background knowledge file not found: {filepath}")
                knowledge[key] = ""
                
        return knowledge
    
    def form_memory(self, user_message: str, emma_response: str, emotional_state: Dict, 
                   biological_state: Dict, context_triggers: List[str] = None) -> Dict:
        """Form a new memory with enhanced emotional and contextual tagging"""
        self.memory_id_counter += 1
        
        # Calculate emotional intensity
        emotional_intensity = sum(abs(v) for v in emotional_state.values()) / len(emotional_state)
        
        # Identify emotional patterns
        dominant_emotion = max(emotional_state.items(), key=lambda x: abs(x[1]))
        
        # Check for background knowledge triggers
        background_triggers = self._identify_background_triggers(user_message, emma_response)
        
        memory = {
            'id': self.memory_id_counter,
            'timestamp': time.time(),
            'user_message': user_message,
            'emma_response': emma_response,
            'emotional_state': emotional_state.copy(),
            'biological_state': biological_state.copy(),
            'emotional_intensity': emotional_intensity,
            'dominant_emotion': dominant_emotion,
            'background_triggers': background_triggers,
            'context_triggers': context_triggers or [],
            'recall_count': 0,
            'importance_score': self._calculate_importance(
                emotional_intensity, background_triggers, context_triggers
            )
        }
        
        self.memories.append(memory)
        self._update_emotional_patterns(memory)
        
        logger.info(f"Formed memory {memory['id']} with importance {memory['importance_score']:.2f}")
        return memory
    
    def _identify_background_triggers(self, user_message: str, emma_response: str) -> List[str]:
        """Identify which background knowledge areas were triggered"""
        triggers = []
        combined_text = (user_message + " " + emma_response).lower()
        
        # Check for key themes from background knowledge
        trigger_keywords = {
            'work_identity': ['work', 'job', 'brand', 'strategy', 'tech', 'company', 'engineer'],
            'childhood': ['family', 'parents', 'brother', 'portland', 'growing up', 'childhood'],
            'fears': ['afraid', 'scared', 'worry', 'anxious', 'elevator', 'phone', 'authentic'],
            'values': ['authentic', 'real', 'honest', 'truth', 'connection', 'meaningful'],
            'relationships': ['friend', 'close', 'trust', 'intimate', 'vulnerable', 'open'],
            'appearance': ['look', 'style', 'hair', 'glasses', 'fashion'],
            'personality': ['curious', 'analyze', 'pattern', 'overthink', 'empathy']
        }
        
        for category, keywords in trigger_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                triggers.append(category)
                
        return triggers
    
    def _calculate_importance(self, emotional_intensity: float, 
                           background_triggers: List[str], context_triggers: List[str]) -> float:
        """Calculate memory importance score"""
        base_score = emotional_intensity
        
        # Boost for background knowledge integration
        if background_triggers:
            base_score += len(background_triggers) * 0.2
            
        # Extra boost for core identity triggers
        if 'work_identity' in background_triggers or 'values' in background_triggers:
            base_score += 0.3
            
        # Boost for personal/intimate topics
        if 'fears' in background_triggers or 'relationships' in background_triggers:
            base_score += 0.25
            
        return min(base_score, 1.0)
    
    def _update_emotional_patterns(self, memory: Dict):
        """Track emotional patterns for better recall"""
        emotion, intensity = memory['dominant_emotion']
        
        if emotion not in self.emotional_patterns:
            self.emotional_patterns[emotion] = []
            
        self.emotional_patterns[emotion].append({
            'memory_id': memory['id'],
            'intensity': intensity,
            'triggers': memory['background_triggers']
        })
    
    def recall_memories(self, query: str, current_emotional_state: Dict, 
                       max_memories: int = 3) -> List[Dict]:
        """Enhanced memory recall with emotional and contextual matching"""
        if not self.memories:
            return []
            
        scored_memories = []
        query_lower = query.lower()
        current_dominant_emotion = max(current_emotional_state.items(), key=lambda x: abs(x[1]))
        
        for memory in self.memories:
            score = 0.0
            
            # Content similarity
            if any(word in memory['user_message'].lower() for word in query_lower.split()):
                score += 0.4
            if any(word in memory['emma_response'].lower() for word in query_lower.split()):
                score += 0.3
                
            # Emotional state similarity
            if memory['dominant_emotion'][0] == current_dominant_emotion[0]:
                score += 0.3
                
            # Background knowledge triggers
            query_triggers = self._identify_background_triggers(query, "")
            common_triggers = set(memory['background_triggers']) & set(query_triggers)
            score += len(common_triggers) * 0.2
            
            # Importance and recency
            score += memory['importance_score'] * 0.2
            days_old = (time.time() - memory['timestamp']) / (24 * 3600)
            score += max(0, (30 - days_old) / 30) * 0.1  # Decay over 30 days
            
            # Boost for rarely recalled memories
            if memory['recall_count'] < 3:
                score += 0.1
                
            scored_memories.append((score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        recalled = [memory for score, memory in scored_memories[:max_memories] if score > 0.2]
        
        # Update recall counts
        for memory in recalled:
            memory['recall_count'] += 1
            
        logger.info(f"Recalled {len(recalled)} memories for query: {query[:50]}...")
        return recalled

class EnhancedDLSController:
    """Enhanced DLS controller with memory and background integration"""
    
    def __init__(self, gemini_api_key: str):
        self.emma = EmmaCompanion(gemini_api_key=gemini_api_key)
        self.memory_system = EnhancedMemorySystem()
        self.conversation_context = []
        
    def process_enhanced_message(self, user_message: str) -> Dict[str, Any]:
        """Process message with enhanced DLS features"""
        start_time = time.time()
        
        # Recall relevant memories
        current_emotional_state = self.emma.dls.get_state_summary()['emotional']
        recalled_memories = self.memory_system.recall_memories(
            user_message, current_emotional_state
        )
        
        # Get Emma's response
        response = self.emma.process_message(user_message)
        
        # Get post-response DLS state
        post_state = self.emma.dls.get_state_summary()
        
        # Form new memory
        new_memory = self.memory_system.form_memory(
            user_message=user_message,
            emma_response=response,
            emotional_state=post_state['emotional'],
            biological_state=post_state['biological']
        )
        
        # Update conversation context
        self.conversation_context.append({
            'user_message': user_message,
            'emma_response': response,
            'memories_recalled': len(recalled_memories),
            'memory_formed': new_memory['id'],
            'emotional_intensity': new_memory['emotional_intensity'],
            'timestamp': time.time()
        })
        
        # Keep only recent context
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)
            
        processing_time = time.time() - start_time
        
        return {
            'response': response,
            'memories_recalled': recalled_memories,
            'new_memory': new_memory,
            'dls_state': post_state,
            'processing_time': processing_time,
            'background_triggers': new_memory['background_triggers']
        }
    
    def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        memories = self.memory_system.memories
        
        if not memories:
            return {'total_memories': 0}
            
        avg_importance = sum(m['importance_score'] for m in memories) / len(memories)
        emotional_distribution = {}
        
        for memory in memories:
            emotion = memory['dominant_emotion'][0]
            emotional_distribution[emotion] = emotional_distribution.get(emotion, 0) + 1
            
        return {
            'total_memories': len(memories),
            'avg_importance': avg_importance,
            'emotional_distribution': emotional_distribution,
            'background_triggers_used': sum(len(m['background_triggers']) for m in memories),
            'most_recalled': max(memories, key=lambda x: x['recall_count']) if memories else None
        }

def test_enhanced_dls():
    """Test the enhanced DLS system"""
    print("Testing Enhanced DLS with Memory and Background Integration")
    print("=" * 60)
    
    # Initialize enhanced system
    gemini_api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyCTbfBx83ffBK5s12SjUbANcZ_jhKw0eDk')
    enhanced_dls = EnhancedDLSController(gemini_api_key)
    
    # Test messages that trigger different background knowledge
    test_messages = [
        "Tell me about your work in brand strategy",
        "What was your childhood like?", 
        "What are you afraid of?",
        "Do you remember our conversation about work?",  # Memory recall test
        "How do you handle vulnerability in relationships?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Testing: {message}")
        print("-" * 40)
        
        result = enhanced_dls.process_enhanced_message(message)
        
        print(f"Response: {result['response']}")
        print(f"Memories recalled: {len(result['memories_recalled'])}")
        print(f"Background triggers: {result['background_triggers']}")
        print(f"Memory importance: {result['new_memory']['importance_score']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        time.sleep(1)  # Brief pause between messages
    
    # Show memory stats
    print(f"\nMemory System Stats:")
    stats = enhanced_dls.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_enhanced_dls()
