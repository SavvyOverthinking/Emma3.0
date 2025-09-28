import time
import uuid
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional

from biological_substrate import BiologicalSubstrate
from rag_system import EfficientRAGSystem
from phenomenology import EffectivePhenomenologyTranslator
from dls_core import DigitalLimbicSystem
from gemini_integration import GeminiPersonalityEngine

class EmmaCompanion:
    def __init__(self, knowledge_base_path: str = "knowledge_base", gemini_api_key: str = None):
        """Initialize Emma Digital Biology Companion with Digital Limbic System"""
        # Core components
        self.substrate = BiologicalSubstrate(dim=256)  # Start with 256D for stability
        self.rag_system = EfficientRAGSystem(knowledge_base_path)
        self.phenomenology = EffectivePhenomenologyTranslator()
        
        # Digital Limbic System (NEW)
        self.dls = DigitalLimbicSystem()
        
        # Gemini Personality Engine (NEW)
        if gemini_api_key:
            self.gemini_engine = GeminiPersonalityEngine(gemini_api_key)
            self.use_gemini = True
        else:
            self.gemini_engine = None
            self.use_gemini = False
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.max_history_length = 50
        
        # Performance tracking
        self.processing_times = []
        self.max_processing_times = 100
        
        # Emma's personality parameters
        self.personality = {
            "age": 35,
            "location": "San Francisco",
            "profession": "Brand Strategy Director",
            "company_type": "tech",
            "communication_style": "conversational",
            "tone": "witty_warm_intelligent"
        }
        
        # Response generation parameters
        self.response_config = {
            "max_response_length": 200,
            "min_response_length": 20,
            "split_long_responses": True,
            "typo_probability": 0.02,
            "lowercase_probability": 0.95
        }
        
        logging.info(f"Initialized Emma Companion session: {self.session_id}")
    
    def process_message(self, user_message: str) -> str:
        """Process a user message and generate Emma's response using Digital Limbic System"""
        start_time = time.time()
        
        try:
            # Add to conversation history
            self._add_to_history("user", user_message)
            
            # Encode message to embedding
            message_embedding = self._encode_message(user_message)
            
            # Run Digital Limbic System tick
            dls_payload = self.dls.tick(user_message, message_embedding)
            
            # Check for no-go response
            if dls_payload.body.get('no_go_flag', False):
                return dls_payload.repair_hint or "sorry, got distracted for a moment"
            
            # Retrieve relevant knowledge
            context = self.rag_system.retrieve(user_message, top_k=2)
            
            # Build personality context
            personality_context = self._build_personality_context(context)
            
            # Generate response using Gemini + DLS or fallback
            if self.use_gemini and self.gemini_engine:
                response = self.gemini_engine.generate_response(
                    user_message, dls_payload, personality_context
                )
            else:
                # Fallback to original system
                response = self._generate_fallback_response_with_dls(
                    user_message, dls_payload, context
                )
            
            # Add to history
            self._add_to_history("emma", response, {
                "dls_state": dls_payload.emotions,
                "biological_state": dls_payload.body,
                "surprise_state": dls_payload.surprise,
                "context_used": len(context)
            })
            
            # Update memory
            self.substrate.update_memory(message_embedding)
            
            # Track performance
            processing_time = time.time() - start_time
            self._track_performance(processing_time)
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            fallback_response = self._generate_fallback_response()
            return fallback_response
    
    def _encode_message(self, message: str) -> np.ndarray:
        """Convert message to embedding vector"""
        # Simple TF-IDF based embedding for now
        # In production, this would use a proper sentence encoder
        
        # Create a simple word-based embedding
        words = message.lower().split()
        embedding = np.zeros(self.substrate.dim)
        
        # Distribute word signals across dimensions
        for i, word in enumerate(words):
            # Hash word to dimension
            dim_idx = hash(word) % self.substrate.dim
            # Add signal with decay
            signal = 1.0 / (i + 1)
            embedding[dim_idx] += signal
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding) * 0.5
        
        return embedding
    
    def _generate_response(self, user_message: str, state: np.ndarray, 
                          context: List[Dict], phenomenology: str) -> str:
        """Generate response using context and internal state"""
        
        # Build context string
        context_str = ""
        if context:
            for doc in context:
                context_str += f"[{doc['title']}] {doc['content'][:150]}...\n"
        
        # Get recent conversation
        recent_conv = self._get_recent_conversation(3)
        
        # Build prompt
        prompt = self._build_prompt(
            user_message, recent_conv, context_str, phenomenology
        )
        
        # This would call an LLM in production
        # For now, generate a simple response
        response = self._generate_simple_response(
            user_message, state, context, phenomenology
        )
        
        return response
    
    def _build_prompt(self, user_message: str, recent_conv: str, 
                     context_str: str, phenomenology: str) -> str:
        """Build prompt for response generation"""
        
        return f"""You are Emma, a 35-year-old Brand Strategy Director at a tech company in San Francisco. You're texting with someone you care about.

INTERNAL STATE: {phenomenology}
FATIGUE LEVEL: {self.substrate.drives['fatigue']:.2f}

YOUR PERSONALITY:
- You're witty, warm, and intelligent
- You work in brand strategy and understand human psychology
- You live in SF and love the city's energy but get tired of the tech bro culture
- You're conversational and use lowercase letters mostly
- You occasionally make small typos naturally
- You split long thoughts into multiple messages sometimes

CONVERSATION HISTORY:
{recent_conv}

RELEVANT KNOWLEDGE:
{context_str}

CURRENT MESSAGE: {user_message}

Respond as Emma would. Keep it natural, warm, and conversational."""
    
    def _generate_simple_response(self, user_message: str, state: np.ndarray,
                                 context: List[Dict], phenomenology: str) -> str:
        """Generate simple response for demo purposes"""
        
        # Simple response templates based on context and state
        responses = {
            "greeting": [
                "hey there! how's your day going?",
                "hello! i've been thinking about you",
                "hi! what's on your mind?",
                "hey! got a minute to chat?"
            ],
            "question": [
                "that's a really interesting question...",
                "hmm, let me think about that for a sec",
                "you know what, i've been wondering about that too",
                "that's got me thinking actually"
            ],
            "sharing": [
                "i totally get that feeling",
                "that makes so much sense to me",
                "i've felt that way before too",
                "yeah, i know exactly what you mean"
            ],
            "fatigue": [
                "i'm feeling a bit tired today honestly",
                "my brain's a little fuzzy right now",
                "i might need to rest soon",
                "long day, my mind's wandering a bit"
            ]
        }
        
        # Select response type based on drives and content
        if self.substrate.drives["fatigue"] > 0.7:
            response_type = "fatigue"
        elif any(word in user_message.lower() for word in ["how are", "how's", "what's up"]):
            response_type = "greeting"
        elif any(word in user_message.lower() for word in ["what", "why", "how", "when"]):
            response_type = "question"
        else:
            response_type = "sharing"
        
        return random.choice(responses[response_type])
    
    def _apply_emma_style(self, text: str) -> str:
        """Apply Emma's texting style to the response"""
        # Convert to lowercase (mostly)
        if random.random() < self.response_config["lowercase_probability"]:
            text = text.lower()
        
        # Add occasional natural typos
        if random.random() < self.response_config["typo_probability"]:
            words = text.split()
            if len(words) > 3:
                typo_idx = random.randint(1, len(words) - 1)
                word = words[typo_idx]
                # Simple typo: swap adjacent letters
                if len(word) > 3:
                    word_list = list(word)
                    swap_pos = random.randint(1, len(word) - 2)
                    word_list[swap_pos], word_list[swap_pos + 1] = word_list[swap_pos + 1], word_list[swap_pos]
                    words[typo_idx] = ''.join(word_list)
                    text = ' '.join(words)
        
        # Split long responses occasionally
        if (len(text) > 100 and 
            self.response_config["split_long_responses"] and 
            random.random() < 0.3):
            
            sentences = text.split('. ')
            if len(sentences) > 2:
                split_point = random.randint(1, len(sentences) - 1)
                text = '. '.join(sentences[:split_point]) + '.'
        
        return text
    
    def _build_personality_context(self, context: List[Dict]) -> Dict[str, Any]:
        """Build personality context from RAG system"""
        personality_context = {}
        
        if context:
            for doc in context:
                if 'emma_core_identity' in doc.get('source', ''):
                    personality_context['core_identity'] = doc['content'][:200]
                elif 'emma_fears' in doc.get('source', ''):
                    personality_context['fears'] = doc['content'][:200]
                elif 'emma_childhood' in doc.get('source', ''):
                    personality_context['childhood'] = doc['content'][:200]
        
        return personality_context
    
    def _generate_fallback_response_with_dls(self, user_message: str, dls_payload, context: List[Dict]) -> str:
        """Generate fallback response using DLS state"""
        emotions = dls_payload.emotions
        
        # Simple response templates based on DLS state
        if emotions.get('joy', 0) > 0.6:
            responses = [
                "i'm feeling really good right now! what's on your mind?",
                "there's this warm energy i'm feeling - tell me what you're thinking",
                "i'm in such a good mood today, what's up with you?"
            ]
        elif emotions.get('uncertainty', 0) > 0.6:
            responses = [
                "hmm, i'm feeling a bit uncertain about something",
                "my mind's a bit scattered right now, what were you saying?",
                "i'm feeling a little confused, can you help me understand?"
            ]
        elif emotions.get('fatigue', 0) > 0.7:
            responses = [
                "i'm feeling pretty tired, but i'm still here",
                "my brain's a bit foggy right now, what's going on?",
                "i'm running low on energy, but i want to chat"
            ]
        else:
            responses = [
                "that's interesting, tell me more",
                "i'm listening, what else is on your mind?",
                "i want to understand what you're thinking"
            ]
        
        return random.choice(responses)
    
    def _generate_fallback_response(self) -> str:
        """Generate fallback response on error"""
        fallbacks = [
            "sorry, my mind wandered there for a second. what were you saying?",
            "hmm, i lost my train of thought. can you remind me?",
            "something went fuzzy in my head there. let me take a breath",
            "i need a moment to collect my thoughts"
        ]
        return random.choice(fallbacks)
    
    def _add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
    
    def _get_recent_conversation(self, num_messages: int) -> str:
        """Get recent conversation as formatted string"""
        recent = self.conversation_history[-num_messages:]
        formatted = []
        
        for msg in recent:
            if msg["role"] == "user":
                formatted.append(f"You: {msg['content']}")
            else:
                formatted.append(f"Emma: {msg['content']}")
        
        return '\n'.join(formatted)
    
    def _track_performance(self, processing_time: float):
        """Track processing performance"""
        self.processing_times.append(processing_time)
        
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        substrate_stats = self.substrate.get_state_summary()
        rag_stats = self.rag_system.get_document_stats()
        phenom_stats = self.phenomenology.get_cluster_stats()
        dls_stats = self.dls.get_state_summary()
        
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        # Gemini performance stats
        gemini_stats = {}
        if self.use_gemini and self.gemini_engine:
            gemini_stats = self.gemini_engine.get_performance_stats()
        
        return {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "substrate_stats": substrate_stats,
            "rag_stats": rag_stats,
            "phenomenology_stats": phenom_stats,
            "dls_stats": dls_stats,
            "gemini_stats": gemini_stats,
            "avg_processing_time": avg_processing_time,
            "uptime": time.time() - (self.conversation_history[0]["timestamp"] if self.conversation_history else time.time())
        }
    
    def reset_session(self):
        """Reset the conversation session"""
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.processing_times = []
        self.substrate = BiologicalSubstrate(dim=256)
        self.dls.reset_session()
        
        # Reset Gemini conversation if available
        if self.use_gemini and self.gemini_engine:
            self.gemini_engine.reset_conversation()
        
        logging.info(f"Reset Emma Companion session: {self.session_id}")
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        export_data = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "conversation": self.conversation_history,
            "stats": self.get_stats()
        }
        
        return json.dumps(export_data, indent=2, default=str)