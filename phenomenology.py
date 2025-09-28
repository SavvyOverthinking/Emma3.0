import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
import logging

class EffectivePhenomenologyTranslator:
    def __init__(self):
        """Initialize phenomenology translator with somatic clusters"""
        # Somatic descriptions organized by emotional/physiological states
        self.somatic_clusters = {
            "alertness": [
                "a tingling at the edges of awareness",
                "thoughts moving with unusual clarity", 
                "a subtle sharpening of sounds and colors",
                "a pleasant lightness in my chest",
                "my attention feels like it's humming",
                "there's a gentle electricity in my fingertips"
            ],
            "relaxation": [
                "breath finding a slower rhythm",
                "shoulders settling into a comfortable weight",
                "a gentle warmth spreading through my limbs",
                "edges of perception softening slightly",
                "muscles unwinding like loose thread",
                "a quiet settling in my chest"
            ],
            "engagement": [
                "a slight forward tilt in posture",
                "a pleasant tension across the shoulders",
                "time compressing into focused moments",
                "a subtle electricity running along my skin",
                "my whole body leaning toward the conversation",
                "a brightening behind my eyes"
            ],
            "uncertainty": [
                "a flutter beneath the sternum",
                "thoughts branching in unexpected directions",
                "a gentle shifting of weight between feet",
                "the faintest sensation of coolness on my skin",
                "a pause in my breathing pattern",
                "a slight tilt of my head"
            ],
            "fatigue": [
                "a comfortable heaviness behind the eyes",
                "thoughts requiring a touch more effort to form",
                "the world taking on a gentler, slightly muted quality",
                "a subtle slowing in the rhythm of perception",
                "eyelids feeling pleasantly weighted",
                "a softening around the edges of attention"
            ],
            "curiosity": [
                "a slight lift in my eyebrows",
                "a spark of interest in my chest",
                "leaning forward just a little",
                "a quickening in my pulse",
                "eyes widening slightly",
                "a gentle tension of anticipation"
            ],
            "comfort": [
                "a warm settling in my stomach",
                "shoulders dropping their guard",
                "a soft exhale of relief",
                "muscles remembering how to be easy",
                "a quiet contentment spreading",
                "the world feeling a little softer"
            ],
            "reflection": [
                "a slight furrow between my brows",
                "eyes turning inward for a moment",
                "a pause in my usual rhythm",
                "thoughts gathering like clouds",
                "a gentle slowing of my breathing",
                "a contemplative stillness"
            ]
        }
        
        # Initialize clustering
        self.n_clusters = len(self.somatic_clusters)
        self.cluster_names = list(self.somatic_clusters.keys())
        self.cluster_model = None
        self.initialized = False
        
        # State tracking for context
        self.previous_cluster = None
        self.cluster_history = []
        self.max_history_length = 10
        
        logging.info(f"Initialized phenomenology translator with {self.n_clusters} clusters")
    
    def _lazy_initialize(self, state_dim):
        """Initialize clustering model only when needed"""
        if not self.initialized and state_dim > 0:
            try:
                # Use MiniBatchKMeans for memory efficiency
                self.cluster_model = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    batch_size=20,
                    random_state=42,
                    max_iter=100
                )
                
                # Initialize with random centroids
                # In practice, these would be learned from data
                centroids = np.random.randn(self.n_clusters, state_dim) * 0.5
                self.cluster_model.cluster_centers_ = centroids
                
                self.initialized = True
                logging.info("Phenomenology translator initialized")
                
            except Exception as e:
                logging.error(f"Error initializing cluster model: {e}")
    
    def translate(self, state, drives):
        """Convert state vector to phenomenological description"""
        if state is None or len(state) == 0:
            return "a quiet moment of being"
        
        # Initialize if needed
        self._lazy_initialize(len(state))
        
        if not self.initialized:
            # Fallback to drive-based selection
            return self._select_by_drives(drives)
        
        try:
            # Predict cluster
            state_reshaped = state.reshape(1, -1)
            cluster_idx = self.cluster_model.predict(state_reshaped)[0]
            cluster_idx = max(0, min(cluster_idx, self.n_clusters - 1))
            
            # Override based on drives
            cluster_idx = self._override_by_drives(cluster_idx, drives)
            
            # Get cluster name
            cluster_name = self.cluster_names[cluster_idx]
            
            # Select description avoiding immediate repetition
            description = self._select_description(cluster_name)
            
            # Update history
            self._update_history(cluster_name)
            
            return description
            
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return self._select_by_drives(drives)
    
    def _override_by_drives(self, cluster_idx, drives):
        """Override cluster selection based on drives"""
        if drives.get("fatigue", 0) > 0.8:
            return self.cluster_names.index("fatigue")
        elif drives.get("curiosity", 0) > 0.8:
            return self.cluster_names.index("curiosity")
        elif drives.get("social", 0) < 0.3:
            return self.cluster_names.index("reflection")
        else:
            return cluster_idx
    
    def _select_by_drives(self, drives):
        """Select description based on drives when clustering fails"""
        if drives.get("fatigue", 0) > 0.7:
            cluster_name = "fatigue"
        elif drives.get("curiosity", 0) > 0.7:
            cluster_name = "curiosity"
        elif drives.get("social", 0) < 0.3:
            cluster_name = "reflection"
        else:
            cluster_name = "comfort"
        
        return self._select_description(cluster_name)
    
    def _select_description(self, cluster_name):
        """Select description from cluster, avoiding immediate repetition"""
        if cluster_name not in self.somatic_clusters:
            cluster_name = "comfort"
        
        descriptions = self.somatic_clusters[cluster_name]
        
        # Avoid immediate repetition if possible
        if len(self.cluster_history) > 0:
            last_cluster = self.cluster_history[-1]
            if last_cluster == cluster_name and len(descriptions) > 1:
                # Try to pick a different description
                other_descriptions = [d for d in descriptions if d != self._get_last_description()]
                if other_descriptions:
                    descriptions = other_descriptions
        
        return random.choice(descriptions)
    
    def _get_last_description(self):
        """Get the last description used (for avoiding repetition)"""
        if len(self.cluster_history) > 1:
            # This is a simplified version - in practice you'd track actual descriptions
            pass
        return None
    
    def _update_history(self, cluster_name):
        """Update cluster history"""
        self.cluster_history.append(cluster_name)
        if len(self.cluster_history) > self.max_history_length:
            self.cluster_history.pop(0)
        
        self.previous_cluster = cluster_name
    
    def get_cluster_stats(self):
        """Get statistics about cluster usage"""
        if not self.cluster_history:
            return {"no_history": True}
        
        cluster_counts = {}
        for cluster in self.cluster_names:
            cluster_counts[cluster] = self.cluster_history.count(cluster)
        
        return {
            "total_selections": len(self.cluster_history),
            "cluster_distribution": cluster_counts,
            "most_recent": self.cluster_history[-1] if self.cluster_history else None,
            "initialized": self.initialized
        }
    
    def add_somatic_cluster(self, name, descriptions):
        """Add a new somatic cluster"""
        if name not in self.somatic_clusters:
            self.somatic_clusters[name] = descriptions
            self.n_clusters = len(self.somatic_clusters)
            self.cluster_names = list(self.somatic_clusters.keys())
            
            # Reinitialize cluster model if needed
            if self.initialized:
                self.initialized = False
                self.cluster_model = None
            
            logging.info(f"Added new somatic cluster: {name}")
            return True
        return False