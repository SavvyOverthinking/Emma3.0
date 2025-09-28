#!/usr/bin/env python3
"""
Memory Manager for Emma's Digital Limbic System
Handles memory pruning, consolidation, and optimization for production stability
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple
from scipy import sparse
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management with pruning and consolidation"""
    
    def __init__(self, max_memories=10000, pruning_threshold=0.01, 
                 consolidation_interval=100, weak_memory_threshold=0.3):
        self.max_memories = max_memories
        self.pruning_threshold = pruning_threshold
        self.consolidation_interval = consolidation_interval
        self.weak_memory_threshold = weak_memory_threshold
        self.last_consolidation = 0
        self.memory_stats = {
            'total_formed': 0,
            'total_pruned': 0,
            'consolidation_runs': 0,
            'avg_importance': 0.0
        }
        
        logger.info(f"MemoryManager initialized: max={max_memories}, threshold={pruning_threshold}")
    
    def should_form_memory(self, emotional_intensity: float, importance_score: float) -> bool:
        """Decide if a memory should be formed based on system capacity and importance"""
        # Always form high-importance memories
        if importance_score > 0.8:
            return True
            
        # Form medium-importance memories if we have capacity
        if importance_score > 0.5 and self.memory_stats['total_formed'] < self.max_memories * 0.8:
            return True
            
        # Only form low-importance memories if we have lots of capacity
        if importance_score > 0.3 and self.memory_stats['total_formed'] < self.max_memories * 0.5:
            return True
            
        return False
    
    def prune_weak_connections(self, memory_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Remove weak connections below threshold to optimize memory"""
        if memory_matrix is None:
            return None
            
        original_nnz = memory_matrix.nnz
        
        # Remove connections below threshold
        memory_matrix.data[np.abs(memory_matrix.data) < self.pruning_threshold] = 0
        memory_matrix.eliminate_zeros()
        
        pruned_count = original_nnz - memory_matrix.nnz
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} weak connections from memory matrix")
            
        return memory_matrix
    
    def consolidate_memories(self, memories: List[Dict], force: bool = False) -> List[Dict]:
        """Consolidate memories when at capacity, keeping most important"""
        if not force and len(memories) <= self.max_memories:
            return memories
            
        if len(memories) <= self.max_memories * 0.9 and not force:
            return memories
            
        logger.info(f"Starting memory consolidation: {len(memories)} -> {self.max_memories}")
        
        # Calculate composite scores for ranking
        scored_memories = []
        current_time = time.time()
        
        for memory in memories:
            score = self._calculate_consolidation_score(memory, current_time)
            scored_memories.append((score, memory))
        
        # Keep the top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        consolidated = [memory for score, memory in scored_memories[:self.max_memories]]
        
        # Update stats
        pruned_count = len(memories) - len(consolidated)
        self.memory_stats['total_pruned'] += pruned_count
        self.memory_stats['consolidation_runs'] += 1
        self.last_consolidation = current_time
        
        # Update recall counts for kept memories (they survived consolidation)
        for memory in consolidated:
            memory['consolidation_survivals'] = memory.get('consolidation_survivals', 0) + 1
            
        logger.info(f"Memory consolidation complete: pruned {pruned_count} memories")
        return consolidated
    
    def _calculate_consolidation_score(self, memory: Dict, current_time: float) -> float:
        """Calculate composite score for memory consolidation ranking"""
        base_importance = memory.get('importance_score', 0.0)
        recall_count = memory.get('recall_count', 0)
        emotional_intensity = memory.get('emotional_intensity', 0.0)
        
        # Time decay factor (memories decay over time)
        age_hours = (current_time - memory.get('timestamp', current_time)) / 3600
        time_decay = np.exp(-age_hours / (24 * 7))  # Week half-life
        
        # Recall boost (frequently accessed memories are more valuable)
        recall_boost = min(recall_count * 0.1, 0.5)  # Max 50% boost
        
        # Emotional significance
        emotion_boost = emotional_intensity * 0.3
        
        # Background knowledge boost
        bg_trigger_boost = len(memory.get('background_triggers', [])) * 0.1
        
        # VTA golden memory bonus
        golden_bonus = 0.2 if memory.get('vta_golden', False) else 0.0
        
        # Consolidation survival bonus
        survival_bonus = memory.get('consolidation_survivals', 0) * 0.05
        
        composite_score = (
            base_importance * time_decay + 
            recall_boost + 
            emotion_boost + 
            bg_trigger_boost + 
            golden_bonus + 
            survival_bonus
        )
        
        return composite_score
    
    def optimize_memory_access(self, memories: List[Dict]) -> Dict[str, List[Dict]]:
        """Create optimized indices for faster memory access"""
        indices = {
            'by_importance': [],
            'by_emotion': {},
            'by_triggers': {},
            'by_recency': []
        }
        
        # Sort by importance
        indices['by_importance'] = sorted(
            memories, 
            key=lambda x: x.get('importance_score', 0.0), 
            reverse=True
        )
        
        # Group by dominant emotion
        for memory in memories:
            emotion = memory.get('dominant_emotion', ('unknown', 0.0))[0]
            if emotion not in indices['by_emotion']:
                indices['by_emotion'][emotion] = []
            indices['by_emotion'][emotion].append(memory)
        
        # Group by background triggers
        for memory in memories:
            for trigger in memory.get('background_triggers', []):
                if trigger not in indices['by_triggers']:
                    indices['by_triggers'][trigger] = []
                indices['by_triggers'][trigger].append(memory)
        
        # Sort by recency
        indices['by_recency'] = sorted(
            memories,
            key=lambda x: x.get('timestamp', 0.0),
            reverse=True
        )
        
        return indices
    
    def get_memory_health_metrics(self, memories: List[Dict]) -> Dict[str, Any]:
        """Generate memory system health metrics"""
        if not memories:
            return {'status': 'empty', 'total_memories': 0}
            
        # Calculate statistics
        importance_scores = [m.get('importance_score', 0.0) for m in memories]
        recall_counts = [m.get('recall_count', 0) for m in memories]
        ages_hours = [(time.time() - m.get('timestamp', time.time())) / 3600 for m in memories]
        
        # Emotional distribution
        emotion_dist = {}
        for memory in memories:
            emotion = memory.get('dominant_emotion', ('unknown', 0.0))[0]
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # Background trigger usage
        trigger_usage = {}
        for memory in memories:
            for trigger in memory.get('background_triggers', []):
                trigger_usage[trigger] = trigger_usage.get(trigger, 0) + 1
        
        metrics = {
            'status': 'healthy' if len(memories) < self.max_memories * 0.9 else 'near_capacity',
            'total_memories': len(memories),
            'capacity_utilization': len(memories) / self.max_memories,
            'avg_importance': np.mean(importance_scores),
            'max_importance': np.max(importance_scores),
            'avg_recall_count': np.mean(recall_counts),
            'avg_age_hours': np.mean(ages_hours),
            'oldest_memory_hours': np.max(ages_hours) if ages_hours else 0,
            'emotion_distribution': emotion_dist,
            'top_triggers': sorted(trigger_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            'consolidation_stats': self.memory_stats,
            'needs_consolidation': len(memories) > self.max_memories * 0.85
        }
        
        return metrics
    
    def suggest_memory_optimizations(self, metrics: Dict[str, Any]) -> List[str]:
        """Suggest optimizations based on memory health metrics"""
        suggestions = []
        
        if metrics['capacity_utilization'] > 0.9:
            suggestions.append("URGENT: Memory near capacity - run consolidation immediately")
        elif metrics['capacity_utilization'] > 0.8:
            suggestions.append("WARNING: Memory usage high - consider consolidation soon")
            
        if metrics['avg_importance'] < 0.5:
            suggestions.append("Many low-importance memories - increase formation threshold")
            
        if metrics['avg_recall_count'] < 0.5:
            suggestions.append("Many memories never recalled - consider more aggressive pruning")
            
        if metrics['oldest_memory_hours'] > 24 * 30:  # 30 days
            suggestions.append("Very old memories present - consider time-based pruning")
            
        # Check for emotional imbalance
        emotion_dist = metrics['emotion_distribution']
        if len(emotion_dist) > 0:
            max_emotion_pct = max(emotion_dist.values()) / sum(emotion_dist.values())
            if max_emotion_pct > 0.7:
                suggestions.append("Emotional imbalance detected - one emotion dominates memories")
        
        if not suggestions:
            suggestions.append("Memory system operating optimally")
            
        return suggestions

class MemoryConnectionGraph:
    """Manages connections between memories for contextual recall"""
    
    def __init__(self, decay_rate=0.01):
        self.connections = {}  # memory_id -> {connected_id: strength}
        self.decay_rate = decay_rate
        
    def add_connection(self, memory_id_1: int, memory_id_2: int, strength: float = 1.0):
        """Add or strengthen connection between memories"""
        if memory_id_1 not in self.connections:
            self.connections[memory_id_1] = {}
        if memory_id_2 not in self.connections:
            self.connections[memory_id_2] = {}
            
        # Bidirectional connection
        self.connections[memory_id_1][memory_id_2] = strength
        self.connections[memory_id_2][memory_id_1] = strength
    
    def strengthen_connection(self, memory_id_1: int, memory_id_2: int, boost: float = 0.1):
        """Strengthen existing connection"""
        if (memory_id_1 in self.connections and 
            memory_id_2 in self.connections[memory_id_1]):
            current = self.connections[memory_id_1][memory_id_2]
            new_strength = min(current + boost, 1.0)
            self.add_connection(memory_id_1, memory_id_2, new_strength)
    
    def decay_connections(self):
        """Apply decay to all connections"""
        for memory_id in self.connections:
            for connected_id in list(self.connections[memory_id].keys()):
                current_strength = self.connections[memory_id][connected_id]
                new_strength = current_strength * (1 - self.decay_rate)
                
                if new_strength < 0.1:  # Remove very weak connections
                    del self.connections[memory_id][connected_id]
                else:
                    self.connections[memory_id][connected_id] = new_strength
    
    def get_connected_memories(self, memory_id: int, min_strength: float = 0.2) -> List[Tuple[int, float]]:
        """Get memories connected to given memory above strength threshold"""
        if memory_id not in self.connections:
            return []
            
        connected = [
            (connected_id, strength) 
            for connected_id, strength in self.connections[memory_id].items()
            if strength >= min_strength
        ]
        
        return sorted(connected, key=lambda x: x[1], reverse=True)

def test_memory_manager():
    """Test memory manager functionality"""
    print("Testing Memory Manager")
    print("=" * 50)
    
    # Create test memories
    manager = MemoryManager(max_memories=5)  # Small for testing
    
    test_memories = []
    for i in range(10):
        memory = {
            'id': i,
            'timestamp': time.time() - (i * 3600),  # Spread over time
            'importance_score': np.random.uniform(0.1, 1.0),
            'emotional_intensity': np.random.uniform(0.0, 1.0),
            'recall_count': np.random.randint(0, 5),
            'background_triggers': ['work_identity', 'values'][:np.random.randint(0, 3)],
            'content': f"Test memory {i}"
        }
        test_memories.append(memory)
    
    print(f"Created {len(test_memories)} test memories")
    
    # Test consolidation
    consolidated = manager.consolidate_memories(test_memories, force=True)
    print(f"After consolidation: {len(consolidated)} memories")
    
    # Test health metrics
    metrics = manager.get_memory_health_metrics(consolidated)
    print(f"Memory health: {metrics['status']}")
    print(f"Average importance: {metrics['avg_importance']:.2f}")
    
    # Test suggestions
    suggestions = manager.suggest_memory_optimizations(metrics)
    print("Optimization suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")

if __name__ == "__main__":
    test_memory_manager()
