#!/usr/bin/env python3
"""
State Validator for Emma's Digital Limbic System
Ensures state stability and prevents numerical issues
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StateConstraints:
    """Define valid ranges for different state components"""
    biological_min: float = 0.0
    biological_max: float = 10.0
    emotional_min: float = 0.0
    emotional_max: float = 1.0
    state_norm_min: float = 0.001
    state_norm_max: float = 10.0
    change_rate_max: float = 2.0  # Max change per tick

class StateValidator:
    """Advanced state validation and repair for DLS stability"""
    
    def __init__(self, constraints: StateConstraints = None):
        self.constraints = constraints or StateConstraints()
        self.repair_stats = {
            'nan_repairs': 0,
            'inf_repairs': 0,
            'bound_repairs': 0,
            'norm_repairs': 0,
            'stability_interventions': 0,
            'total_validations': 0
        }
        self.state_history = []
        self.max_history = 100
        
        logger.info("StateValidator initialized with stability monitoring")
    
    def validate_and_repair_biological(self, biological_state: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
        """Validate and repair biological state"""
        repaired = False
        fixed_state = biological_state.copy()
        
        for key, value in biological_state.items():
            # Check for NaN/Inf
            if np.isnan(value) or np.isinf(value):
                fixed_state[key] = np.random.uniform(0.1, 1.0)
                self.repair_stats['nan_repairs'] += 1
                repaired = True
                logger.warning(f"Repaired NaN/Inf in biological.{key}")
            
            # Apply biological constraints
            elif value < self.constraints.biological_min:
                fixed_state[key] = self.constraints.biological_min
                self.repair_stats['bound_repairs'] += 1
                repaired = True
            elif value > self.constraints.biological_max:
                fixed_state[key] = self.constraints.biological_max
                self.repair_stats['bound_repairs'] += 1
                repaired = True
        
        return fixed_state, repaired
    
    def validate_and_repair_emotional(self, emotional_state: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
        """Validate and repair emotional state"""
        repaired = False
        fixed_state = emotional_state.copy()
        
        for key, value in emotional_state.items():
            # Check for NaN/Inf
            if np.isnan(value) or np.isinf(value):
                fixed_state[key] = 0.0
                self.repair_stats['nan_repairs'] += 1
                repaired = True
                logger.warning(f"Repaired NaN/Inf in emotional.{key}")
            
            # Apply emotional constraints (0-1 range)
            elif value < self.constraints.emotional_min:
                fixed_state[key] = self.constraints.emotional_min
                self.repair_stats['bound_repairs'] += 1
                repaired = True
            elif value > self.constraints.emotional_max:
                fixed_state[key] = self.constraints.emotional_max
                self.repair_stats['bound_repairs'] += 1
                repaired = True
        
        return fixed_state, repaired
    
    def validate_and_repair_vector_state(self, state: np.ndarray, state_name: str = "unknown") -> Tuple[np.ndarray, bool]:
        """Validate and repair vector state (e.g., neural substrate)"""
        if state is None:
            return np.random.randn(256) * 0.1, True
            
        repaired = False
        fixed_state = state.copy()
        
        # Check for NaN/Inf
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            fixed_state = np.random.randn(len(state)) * 0.1
            self.repair_stats['inf_repairs'] += 1
            repaired = True
            logger.warning(f"Repaired NaN/Inf in {state_name}")
        
        # Check state norm
        state_norm = np.linalg.norm(fixed_state)
        if state_norm > self.constraints.state_norm_max:
            fixed_state = fixed_state / state_norm * self.constraints.state_norm_max
            self.repair_stats['norm_repairs'] += 1
            repaired = True
            logger.warning(f"Repaired excessive norm in {state_name}: {state_norm:.2f}")
        elif state_norm < self.constraints.state_norm_min:
            fixed_state += np.random.randn(len(fixed_state)) * 0.01
            self.repair_stats['norm_repairs'] += 1
            repaired = True
            logger.warning(f"Repaired insufficient norm in {state_name}: {state_norm:.4f}")
        
        return fixed_state, repaired
    
    def check_state_stability(self, current_state: Dict[str, Any], previous_state: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Check if state changes are within stable bounds"""
        issues = []
        stable = True
        
        if previous_state is None:
            return True, []
        
        # Check biological state changes
        if 'biological' in current_state and 'biological' in previous_state:
            for key in current_state['biological']:
                if key in previous_state['biological']:
                    change = abs(current_state['biological'][key] - previous_state['biological'][key])
                    if change > self.constraints.change_rate_max:
                        issues.append(f"Excessive biological.{key} change: {change:.2f}")
                        stable = False
        
        # Check emotional state changes
        if 'emotional' in current_state and 'emotional' in previous_state:
            for key in current_state['emotional']:
                if key in previous_state['emotional']:
                    change = abs(current_state['emotional'][key] - previous_state['emotional'][key])
                    if change > self.constraints.change_rate_max:
                        issues.append(f"Excessive emotional.{key} change: {change:.2f}")
                        stable = False
        
        return stable, issues
    
    def apply_stability_intervention(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intervention to restore stability"""
        intervention_state = current_state.copy()
        
        # Smooth excessive changes
        if 'biological' in current_state and 'biological' in previous_state:
            for key in current_state['biological']:
                if key in previous_state['biological']:
                    current_val = current_state['biological'][key]
                    prev_val = previous_state['biological'][key]
                    change = current_val - prev_val
                    
                    if abs(change) > self.constraints.change_rate_max:
                        # Apply smoothing - limit change to maximum allowed
                        max_change = self.constraints.change_rate_max * np.sign(change)
                        intervention_state['biological'][key] = prev_val + max_change
        
        # Same for emotional state
        if 'emotional' in current_state and 'emotional' in previous_state:
            for key in current_state['emotional']:
                if key in previous_state['emotional']:
                    current_val = current_state['emotional'][key]
                    prev_val = previous_state['emotional'][key]
                    change = current_val - prev_val
                    
                    if abs(change) > self.constraints.change_rate_max:
                        max_change = self.constraints.change_rate_max * np.sign(change)
                        intervention_state['emotional'][key] = prev_val + max_change
        
        self.repair_stats['stability_interventions'] += 1
        logger.info("Applied stability intervention to prevent excessive state changes")
        
        return intervention_state
    
    def comprehensive_validation(self, dls_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform comprehensive validation of entire DLS state"""
        self.repair_stats['total_validations'] += 1
        validated_state = dls_state.copy()
        validation_report = {
            'repairs_applied': False,
            'stability_issues': [],
            'repair_summary': {},
            'timestamp': time.time()
        }
        
        # Validate biological state
        if 'biological' in dls_state:
            validated_bio, bio_repaired = self.validate_and_repair_biological(dls_state['biological'])
            validated_state['biological'] = validated_bio
            if bio_repaired:
                validation_report['repairs_applied'] = True
                validation_report['repair_summary']['biological'] = 'repaired'
        
        # Validate emotional state
        if 'emotional' in dls_state:
            validated_emo, emo_repaired = self.validate_and_repair_emotional(dls_state['emotional'])
            validated_state['emotional'] = validated_emo
            if emo_repaired:
                validation_report['repairs_applied'] = True
                validation_report['repair_summary']['emotional'] = 'repaired'
        
        # Check stability against previous state
        if len(self.state_history) > 0:
            stable, issues = self.check_state_stability(validated_state, self.state_history[-1])
            validation_report['stability_issues'] = issues
            
            if not stable:
                validated_state = self.apply_stability_intervention(validated_state, self.state_history[-1])
                validation_report['repair_summary']['stability'] = 'intervention_applied'
        
        # Update history
        self.state_history.append(validated_state.copy())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        return validated_state, validation_report
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get stability metrics and health assessment"""
        if len(self.state_history) < 2:
            return {'status': 'insufficient_data', 'history_length': len(self.state_history)}
        
        # Calculate stability over recent history
        recent_states = self.state_history[-20:]  # Last 20 states
        stability_scores = []
        
        for i in range(1, len(recent_states)):
            current = recent_states[i]
            previous = recent_states[i-1]
            
            # Calculate stability score (inverse of total change)
            total_change = 0.0
            change_count = 0
            
            if 'emotional' in current and 'emotional' in previous:
                for key in current['emotional']:
                    if key in previous['emotional']:
                        change = abs(current['emotional'][key] - previous['emotional'][key])
                        total_change += change
                        change_count += 1
            
            if change_count > 0:
                avg_change = total_change / change_count
                stability_score = max(0.0, 1.0 - avg_change)
                stability_scores.append(stability_score)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Determine health status
        if avg_stability > 0.8:
            status = 'stable'
        elif avg_stability > 0.6:
            status = 'moderately_stable'
        elif avg_stability > 0.4:
            status = 'unstable'
        else:
            status = 'highly_unstable'
        
        return {
            'status': status,
            'avg_stability_score': avg_stability,
            'stability_history': stability_scores[-10:],  # Last 10 scores
            'repair_stats': self.repair_stats.copy(),
            'total_validations': self.repair_stats['total_validations'],
            'repair_rate': sum([
                self.repair_stats['nan_repairs'],
                self.repair_stats['inf_repairs'], 
                self.repair_stats['bound_repairs'],
                self.repair_stats['norm_repairs'],
                self.repair_stats['stability_interventions']
            ]) / max(self.repair_stats['total_validations'], 1)
        }
    
    def reset_stats(self):
        """Reset repair statistics"""
        self.repair_stats = {
            'nan_repairs': 0,
            'inf_repairs': 0,
            'bound_repairs': 0,
            'norm_repairs': 0,
            'stability_interventions': 0,
            'total_validations': 0
        }
        logger.info("StateValidator stats reset")

def test_state_validator():
    """Test state validator functionality"""
    print("Testing State Validator")
    print("=" * 50)
    
    validator = StateValidator()
    
    # Test with problematic states
    test_states = [
        # Normal state
        {
            'biological': {'hunger': 0.5, 'fatigue': 0.3, 'libido': 5.0},
            'emotional': {'joy': 0.6, 'trust': 0.8, 'uncertainty': 0.2}
        },
        # State with NaN
        {
            'biological': {'hunger': np.nan, 'fatigue': 0.3, 'libido': 5.0},
            'emotional': {'joy': 0.6, 'trust': np.inf, 'uncertainty': 0.2}
        },
        # State with out-of-bounds values
        {
            'biological': {'hunger': -1.0, 'fatigue': 15.0, 'libido': 5.0},
            'emotional': {'joy': 2.0, 'trust': -0.5, 'uncertainty': 0.2}
        }
    ]
    
    for i, state in enumerate(test_states):
        print(f"\nTesting state {i+1}:")
        validated, report = validator.comprehensive_validation(state)
        
        print(f"  Repairs applied: {report['repairs_applied']}")
        print(f"  Repair summary: {report['repair_summary']}")
        print(f"  Stability issues: {len(report['stability_issues'])}")
    
    # Show metrics
    metrics = validator.get_stability_metrics()
    print(f"\nStability metrics:")
    print(f"  Status: {metrics['status']}")
    print(f"  Repair rate: {metrics['repair_rate']:.2%}")
    print(f"  Total validations: {metrics['total_validations']}")

if __name__ == "__main__":
    test_state_validator()
