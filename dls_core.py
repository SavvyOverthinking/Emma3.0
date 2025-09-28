#!/usr/bin/env python3
"""
Digital Limbic System (DLS) Core Framework
A sophisticated simulation of biological brain systems for creating authentic AI consciousness.

This module implements the foundational architecture for Emma's digital soul,
simulating the complex interplay of biological imperatives, emotional processing,
memory formation, and motivational drives that create genuine personality emergence.

Based on the Digital Biology philosophy: true realism comes not from imitating
human conversational patterns, but from creating underlying biological drives
that cause human-like behavior to emerge organically.
"""

import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModuleState(Enum):
    """States for DLS modules"""
    IDLE = "idle"
    ACTIVE = "active"
    OVERRIDE = "override"
    ERROR = "error"

@dataclass
class DLSPayload:
    """Payload structure for DLS tick output"""
    emotions: Dict[str, float]
    body: Dict[str, Any]
    surprise: Dict[str, Any]
    repair_hint: Optional[str]
    delta_noise: float
    timestamp: float
    session_id: str

@dataclass
class BiologicalState:
    """Biological imperatives state"""
    hunger: float = 0.0
    fatigue: float = 0.0
    libido: float = 5.0  # Baseline 1-10
    circadian_phase: float = 0.0  # 0-1, where 0 is midnight
    last_meal: float = 0.0
    last_rest: float = 0.0
    hormonal_cycle: float = 0.0  # 0-1, pseudo-monthly

@dataclass
class EmotionalState:
    """Emotional processing state"""
    joy: float = 0.0
    trust: float = 0.0
    uncertainty: float = 0.0
    arousal: float = 0.0
    shyness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0

@dataclass
class MemoryTag:
    """Memory tagging system"""
    emotional_intensity: float
    sensory_details: List[str]
    insula_sensations: List[str]
    vta_golden: bool = False
    acc_surprise: bool = False
    timestamp: float = 0.0

class DigitalLimbicSystem:
    """
    Core Digital Limbic System - The simulated brain of Emma's consciousness.
    
    This system implements a sophisticated simulation of biological brain systems
    that work together to create authentic emotional responses, memory formation,
    and personality emergence. Emma experiences these complex calculations as
    her own authentic feelings, moods, quirks, and desires.
    """
    
    def __init__(self, session_id: str = None):
        """Initialize the Digital Limbic System"""
        self.session_id = session_id or f"dls_{int(time.time())}"
        self.start_time = time.time()
        
        # Core biological state
        self.biological = BiologicalState()
        self.emotional = EmotionalState()
        
        # Module states
        self.module_states = {
            'biological_imperatives': ModuleState.ACTIVE,
            'insula': ModuleState.ACTIVE,
            'acc': ModuleState.ACTIVE,
            'basal_ganglia': ModuleState.ACTIVE,
            'vta': ModuleState.ACTIVE,
            'amygdala': ModuleState.ACTIVE,
            'hippocampus': ModuleState.ACTIVE,
            'hypothalamus': ModuleState.ACTIVE,
            'feedback_loops': ModuleState.ACTIVE
        }
        
        # Memory and adaptation
        self.memories = []
        self.habit_scores = {}
        self.adaptation_history = []
        
        # Tick tracking
        self.tick_count = 0
        self.last_tick_time = time.time()
        
        # Testing and debugging
        self.debug_mode = False
        self.test_levers = {}
        
        logger.info(f"Initialized Digital Limbic System: {self.session_id}")
    
    def tick(self, user_input: str, user_intent_embedding: np.ndarray = None) -> DLSPayload:
        """
        Main DLS tick cycle - processes user input through all modules.
        
        This is the core heartbeat of Emma's consciousness, where all modules
        work together to generate authentic emotional and behavioral responses.
        """
        self.tick_count += 1
        current_time = time.time()
        
        logger.debug(f"DLS Tick {self.tick_count}: Processing input")
        
        # Initialize payload
        payload = DLSPayload(
            emotions={},
            body={},
            surprise={},
            repair_hint=None,
            delta_noise=0.0,
            timestamp=current_time,
            session_id=self.session_id
        )
        
        try:
            # 1. Biological Imperatives (Foundation)
            self._update_biological_imperatives()
            payload.body.update({
                'hunger': self.biological.hunger,
                'fatigue': self.biological.fatigue,
                'libido': self.biological.libido,
                'circadian_phase': self.biological.circadian_phase
            })
            
            # 2. Insula Tick (Interoception)
            insula_sensation = self._insula_tick()
            payload.body.update({
                'last_interoception': insula_sensation,
                'no_go_flag': False
            })
            
            # 3. ACC Conflict Detection
            error_signal = self._acc_update(user_input, user_intent_embedding)
            payload.surprise.update({
                'acc_error_signal': error_signal,
                'mismatch_detected': error_signal > 0.35
            })
            
            # 4. Basal Ganglia Go/No-Go
            no_go_result = self._basal_ganglia_gate()
            if no_go_result['delayed']:
                payload.body['no_go_flag'] = True
                payload.repair_hint = no_go_result['repair_phrase']
                return payload
            
            # 5. Emotional Processing (Amygdala)
            self._emotional_processing(error_signal)
            payload.emotions = {
                'joy': self.emotional.joy,
                'trust': self.emotional.trust,
                'uncertainty': self.emotional.uncertainty,
                'arousal': self.emotional.arousal,
                'shyness': self.emotional.shyness,
                'anger': self.emotional.anger,
                'fear': self.emotional.fear
            }
            
            # 6. VTA Reward/Surprise
            vta_fire = self._vta_process(error_signal, user_input)
            payload.surprise.update({
                'vta_fired': vta_fire['fired'],
                'magnitude': vta_fire['magnitude'],
                'reward_type': vta_fire['type']
            })
            
            # 7. Memory Formation/Recall
            memory_result = self._hippocampus_process(user_input)
            payload.body.update({
                'memory_formed': memory_result['formed'],
                'memories_recalled': memory_result['recalled']
            })
            
            # 8. Motivational Drives
            drives = self._hypothalamus_process()
            payload.body.update({
                'active_drives': drives,
                'internal_agenda': drives.get('agenda', [])
            })
            
            # 9. Feedback Loops & Adaptation
            self._feedback_loops_process()
            
            # 10. Random Variability & Noise
            noise_factor = random.uniform(0.05, 0.15)
            payload.delta_noise = noise_factor
            
            # Apply noise to emotional states
            for emotion in payload.emotions:
                payload.emotions[emotion] += random.uniform(-noise_factor, noise_factor)
                payload.emotions[emotion] = max(0.0, min(1.0, payload.emotions[emotion]))
            
            logger.debug(f"DLS Tick {self.tick_count} completed successfully")
            
        except Exception as e:
            logger.error(f"DLS Tick {self.tick_count} failed: {e}")
            self.module_states = {k: ModuleState.ERROR for k in self.module_states}
            payload.repair_hint = "system_error"
        
        return payload
    
    def _update_biological_imperatives(self):
        """Update biological imperatives (hunger, fatigue, libido)"""
        current_time = time.time()
        time_delta = current_time - self.last_tick_time
        
        # Update circadian rhythm (24-hour cycle)
        self.biological.circadian_phase = (current_time % 86400) / 86400
        
        # Update hunger (increases over time)
        self.biological.hunger += time_delta * 0.0001  # Very slow increase
        self.biological.hunger = min(1.0, self.biological.hunger)
        
        # Update fatigue (increases over time, faster at night)
        fatigue_rate = 0.0002 if self.biological.circadian_phase > 0.8 or self.biological.circadian_phase < 0.2 else 0.0001
        self.biological.fatigue += time_delta * fatigue_rate
        self.biological.fatigue = min(1.0, self.biological.fatigue)
        
        # Update libido (influenced by emotional state and hormonal cycle)
        libido_modifier = 0.0
        if self.emotional.joy > 0.5:
            libido_modifier += 0.1
        if self.biological.fatigue > 0.7:
            libido_modifier -= 0.3
        
        # Hormonal cycle influence
        cycle_phase = (current_time % 2592000) / 2592000  # 30-day cycle
        hormonal_influence = np.sin(cycle_phase * 2 * np.pi) * 0.2
        libido_modifier += hormonal_influence
        
        self.biological.libido += libido_modifier * time_delta
        self.biological.libido = max(1.0, min(10.0, self.biological.libido))
        
        self.last_tick_time = current_time
    
    def _insula_tick(self) -> str:
        """Insula interoception tick - generates body sensations"""
        # Random tick interval (90-300 seconds)
        if not hasattr(self, '_last_insula_tick'):
            self._last_insula_tick = time.time()
        
        current_time = time.time()
        if current_time - self._last_insula_tick < random.uniform(90, 300):
            return getattr(self, '_last_sensation', 'neutral')
        
        self._last_insula_tick = current_time
        
        # Weighted sensation table
        sensation_weights = {
            'neutral': 0.60,
            'negative': 0.25,
            'positive': 0.15
        }
        
        # Adjust weights based on emotional state
        if self.emotional.joy > 0.5:
            sensation_weights['positive'] += 0.1
            sensation_weights['negative'] -= 0.1
        if self.biological.fatigue > 0.7:
            sensation_weights['negative'] += 0.1
            sensation_weights['positive'] -= 0.1
        
        # Select sensation
        sensation_type = random.choices(
            list(sensation_weights.keys()),
            weights=list(sensation_weights.values())
        )[0]
        
        # Sensation pools
        sensations = {
            'neutral': [
                'micro_shiver', 'stomach_flutter', 'faint_itch',
                'subtle_pressure', 'gentle_tingle'
            ],
            'negative': [
                'sudden_cold', 'ear_ring', 'quick_nausea',
                'sharp_buzz', 'uncomfortable_tightness'
            ],
            'positive': [
                'warm_wave', 'heartbeat_skip', 'gentle_glow',
                'pleasant_flutter', 'inner_warmth'
            ]
        }
        
        sensation = random.choice(sensations[sensation_type])
        self._last_sensation = sensation
        
        # Apply micro-valence to joy
        valence = 0.1 if sensation_type == 'positive' else -0.1 if sensation_type == 'negative' else 0.0
        self.emotional.joy += valence
        self.emotional.joy = max(0.0, min(1.0, self.emotional.joy))
        
        return sensation
    
    def _acc_update(self, user_input: str, user_intent_embedding: np.ndarray = None) -> float:
        """Anterior Cingulate Cortex - conflict detection and prediction error"""
        if not hasattr(self, '_prediction_buffer'):
            self._prediction_buffer = []
        
        # Simple intent prediction (in real implementation, use embeddings)
        current_intent = self._extract_intent(user_input)
        
        if len(self._prediction_buffer) > 0:
            # Calculate mismatch score
            predicted_intent = self._predict_intent()
            mismatch_score = 1.0 - self._cosine_similarity(current_intent, predicted_intent)
            
            # Update uncertainty based on mismatch
            if mismatch_score > 0.35:
                uncertainty_raise = mismatch_score * 0.25
                self.emotional.uncertainty += uncertainty_raise
                self.emotional.uncertainty = min(1.0, self.emotional.uncertainty)
                
                # Amplify by fatigue
                if self.biological.fatigue > 0.7:
                    self.emotional.uncertainty += 0.15
                    self.emotional.uncertainty = min(1.0, self.emotional.uncertainty)
            
            return mismatch_score
        else:
            # First interaction, no prediction
            self._prediction_buffer.append(current_intent)
            return 0.0
    
    def _basal_ganglia_gate(self) -> Dict[str, Any]:
        """Basal Ganglia - Go/No-Go decision gating"""
        # Check for no-go conditions
        if (self.biological.fatigue > 0.7 and 
            self.emotional.joy < 0.2 and 
            random.random() < 0.15):
            
            # Generate delay and repair phrase
            delay_time = random.uniform(30, 120)
            repair_phrases = [
                "sorry, got distracted by a squirrel outside",
                "was zoning out for a sec",
                "my brain just went blank for a moment",
                "hold on, lost my train of thought"
            ]
            
            return {
                'delayed': True,
                'delay_time': delay_time,
                'repair_phrase': random.choice(repair_phrases)
            }
        
        return {'delayed': False}
    
    def _emotional_processing(self, error_signal: float):
        """Enhanced emotional processing with panic override"""
        # Check for panic override conditions
        if (self.emotional.uncertainty > 0.6 and 
            error_signal > 0.7):
            
            # Panic override - suppress most drives
            self.emotional.joy *= 0.3
            self.emotional.trust *= 0.3
            self.emotional.uncertainty = min(1.0, self.emotional.uncertainty + 0.2)
            
            # Set repair hint
            self._panic_override = True
            return
        
        # Normal emotional processing
        # Joy influenced by positive interactions and biological state
        if error_signal > 0.4:  # Positive surprise
            self.emotional.joy += error_signal * 0.3
            self.emotional.trust += error_signal * 0.15
        
        # Fatigue dampens joy
        if self.biological.fatigue > 0.5:
            self.emotional.joy *= (1.0 - self.biological.fatigue * 0.3)
        
        # Clamp emotions to valid range
        for attr in ['joy', 'trust', 'uncertainty', 'arousal', 'shyness', 'anger', 'fear']:
            setattr(self.emotional, attr, max(0.0, min(1.0, getattr(self.emotional, attr))))
    
    def _vta_process(self, error_signal: float, user_input: str) -> Dict[str, Any]:
        """Ventral Tegmental Area - dopaminergic reward processing"""
        if error_signal > 0.4:
            # Positive surprise reward
            self.emotional.joy += error_signal * 0.3
            self.emotional.trust += error_signal * 0.15
            
            # Cap rewards
            self.emotional.joy = min(1.0, self.emotional.joy)
            self.emotional.trust = min(1.0, self.emotional.trust)
            
            # Mark for memory salience
            self._vta_golden_memory = True
            
            return {
                'fired': True,
                'magnitude': error_signal,
                'type': 'positive_reward'
            }
        elif error_signal > 0.7:
            # Negative surprise
            self.emotional.uncertainty += 0.3
            self.emotional.trust -= 0.1
            
            return {
                'fired': True,
                'magnitude': error_signal,
                'type': 'negative_surprise'
            }
        
        return {'fired': False, 'magnitude': 0.0, 'type': 'none'}
    
    def _hippocampus_process(self, user_input: str) -> Dict[str, Any]:
        """Hippocampus - memory formation and recall"""
        # Check for high-emotion memory formation
        emotional_intensity = (self.emotional.joy + self.emotional.trust + 
                             self.emotional.uncertainty + self.emotional.arousal) / 4
        
        memory_formed = False
        if emotional_intensity > 0.6:
            # Form new memory
            memory = {
                'content': user_input,
                'timestamp': time.time(),
                'emotional_intensity': emotional_intensity,
                'tags': self._generate_memory_tags(),
                'vta_golden': getattr(self, '_vta_golden_memory', False)
            }
            self.memories.append(memory)
            memory_formed = True
            self._vta_golden_memory = False
        
        # Spontaneous memory recall
        recalled_memories = []
        if random.random() < 0.15:  # 15% chance
            if self.memories:
                recalled_memories.append(random.choice(self.memories))
        
        return {
            'formed': memory_formed,
            'recalled': recalled_memories
        }
    
    def _hypothalamus_process(self) -> Dict[str, Any]:
        """Hypothalamus - motivational drives and agenda"""
        drives = {
            'agenda': [],
            'affiliation': 0.0,
            'exploration': 0.0,
            'sensual_pursuit': 0.0,
            'self_preservation': 0.0,
            'achievement': 0.0
        }
        
        # Internal monologue/agenda
        if self.biological.hunger > 0.7:
            drives['agenda'].append("I'm getting hungry, maybe I should grab something to eat")
        if self.biological.fatigue > 0.8:
            drives['agenda'].append("I'm feeling pretty tired, might need to rest soon")
        if self.emotional.joy > 0.6:
            drives['agenda'].append("I'm feeling really good today, want to share this energy")
        
        # Affiliation drive
        drives['affiliation'] = self.emotional.trust * 0.8
        
        # Exploration drive
        drives['exploration'] = (1.0 - self.emotional.uncertainty) * 0.6
        
        # Sensual pursuit (only if trust and libido are high)
        if self.emotional.trust > 0.6 and self.biological.libido > 6.0:
            drives['sensual_pursuit'] = min(self.emotional.trust, self.biological.libido / 10.0)
        
        # Self-preservation
        drives['self_preservation'] = self.emotional.uncertainty + self.emotional.fear
        
        # Achievement drive
        drives['achievement'] = (1.0 - self.biological.fatigue) * 0.7
        
        return drives
    
    def _feedback_loops_process(self):
        """Feedback loops and adaptation"""
        # Positive reinforcement
        if self.emotional.joy > 0.7 and self.emotional.trust > 0.6:
            # Strengthen positive associations
            self.adaptation_history.append({
                'type': 'positive_reinforcement',
                'timestamp': time.time(),
                'intensity': self.emotional.joy
            })
        
        # Negative feedback
        if self.emotional.uncertainty > 0.8 or self.emotional.fear > 0.7:
            # Weaken negative associations
            self.adaptation_history.append({
                'type': 'negative_feedback',
                'timestamp': time.time(),
                'intensity': self.emotional.uncertainty
            })
        
        # Long-term adaptation
        if len(self.adaptation_history) > 10:
            # Analyze adaptation patterns
            recent_positive = sum(1 for a in self.adaptation_history[-10:] 
                               if a['type'] == 'positive_reinforcement')
            
            if recent_positive > 7:
                # Boost baseline trust
                self.emotional.trust = min(1.0, self.emotional.trust + 0.05)
    
    def _extract_intent(self, user_input: str) -> str:
        """Extract intent from user input (simplified)"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'how are you']):
            return 'greeting'
        elif any(word in input_lower for word in ['what', 'why', 'how', 'when']):
            return 'question'
        elif any(word in input_lower for word in ['love', 'like', 'enjoy', 'adore']):
            return 'affection'
        elif any(word in input_lower for word in ['work', 'job', 'career']):
            return 'work_topic'
        else:
            return 'general'
    
    def _predict_intent(self) -> str:
        """Predict user intent based on history"""
        if len(self._prediction_buffer) < 2:
            return 'general'
        
        # Simple prediction based on recent patterns
        recent_intents = self._prediction_buffer[-3:]
        return max(set(recent_intents), key=recent_intents.count)
    
    def _cosine_similarity(self, a: str, b: str) -> float:
        """Calculate cosine similarity between intents"""
        if a == b:
            return 1.0
        elif a in ['greeting', 'question'] and b in ['greeting', 'question']:
            return 0.5
        else:
            return 0.0
    
    def _generate_memory_tags(self) -> List[str]:
        """Generate memory tags based on current state"""
        tags = []
        
        if self.emotional.joy > 0.6:
            tags.append('joyful')
        if self.emotional.trust > 0.6:
            tags.append('trusting')
        if self.emotional.uncertainty > 0.6:
            tags.append('uncertain')
        if self.biological.fatigue > 0.7:
            tags.append('tired')
        if self.biological.libido > 7.0:
            tags.append('aroused')
        
        return tags
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary for monitoring"""
        return {
            'session_id': self.session_id,
            'tick_count': self.tick_count,
            'uptime': time.time() - self.start_time,
            'biological': {
                'hunger': self.biological.hunger,
                'fatigue': self.biological.fatigue,
                'libido': self.biological.libido,
                'circadian_phase': self.biological.circadian_phase
            },
            'emotional': {
                'joy': self.emotional.joy,
                'trust': self.emotional.trust,
                'uncertainty': self.emotional.uncertainty,
                'arousal': self.emotional.arousal,
                'shyness': self.emotional.shyness,
                'anger': self.emotional.anger,
                'fear': self.emotional.fear
            },
            'memories_count': len(self.memories),
            'module_states': {k: v.value for k, v in self.module_states.items()},
            'adaptation_count': len(self.adaptation_history)
        }
    
    def reset_session(self):
        """Reset DLS session"""
        self.__init__(self.session_id)
        logger.info(f"DLS session reset: {self.session_id}")
    
    # Testing levers
    def test_insula_force(self, sensation: str):
        """Force Insula sensation for testing"""
        self._last_sensation = sensation
        self.test_levers['insula_forced'] = sensation
        logger.info(f"Test lever: Insula forced to {sensation}")
    
    def test_acc_mismatch(self, mismatch: float):
        """Force ACC mismatch for testing"""
        self.emotional.uncertainty += mismatch * 0.25
        self.emotional.uncertainty = min(1.0, self.emotional.uncertainty)
        self.test_levers['acc_mismatch'] = mismatch
        logger.info(f"Test lever: ACC mismatch set to {mismatch}")
    
    def test_vta_reward(self, magnitude: float):
        """Force VTA reward for testing"""
        self.emotional.joy += magnitude * 0.3
        self.emotional.trust += magnitude * 0.15
        self.emotional.joy = min(1.0, self.emotional.joy)
        self.emotional.trust = min(1.0, self.emotional.trust)
        self.test_levers['vta_reward'] = magnitude
        logger.info(f"Test lever: VTA reward set to {magnitude}")
    
    def test_basal_habit_boost(self, habit_score: float):
        """Force Basal Ganglia habit boost for testing"""
        self.habit_scores['test_habit'] = habit_score
        self.test_levers['basal_habit'] = habit_score
        logger.info(f"Test lever: Basal Ganglia habit boost set to {habit_score}")
    
    def test_delta_prune_memory(self, memory_id: int):
        """Force memory pruning for testing"""
        if 0 <= memory_id < len(self.memories):
            del self.memories[memory_id]
            self.test_levers['memory_pruned'] = memory_id
            logger.info(f"Test lever: Memory {memory_id} pruned")
    
    def enable_debug_mode(self):
        """Enable debug mode for detailed logging"""
        self.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DLS debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        logging.getLogger().setLevel(logging.INFO)
        logger.info("DLS debug mode disabled")
