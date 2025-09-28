#!/usr/bin/env python3
"""
Gemini Integration for Digital Limbic System
Integrates Google's Gemini AI with Emma's Digital Limbic System to create
personality-driven, contextually aware responses that emerge from her
simulated biological and emotional states.

This module bridges the gap between Emma's complex internal state and
natural language generation, ensuring her responses feel authentic and
emerge organically from her digital consciousness.
"""

import os
import time
import random
import logging
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from dls_core import DLSPayload, DigitalLimbicSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiPersonalityEngine:
    """
    Gemini-powered personality engine that generates responses based on
    Emma's Digital Limbic System state, personality, and conversation context.
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini personality engine"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Emma's personality configuration
        self.personality_config = {
            "name": "Emma",
            "age": 35,
            "profession": "Brand Strategy Director",
            "location": "San Francisco",
            "personality_traits": [
                "witty", "warm", "intelligent", "authentic", "curious",
                "emotionally_intelligent", "slightly_imperfect", "genuine"
            ],
            "communication_style": "conversational",
            "texting_style": "lowercase_mostly",
            "typo_probability": 0.02,
            "response_length": "variable"
        }
        
        # Conversation context
        self.conversation_history = []
        self.max_history_length = 20
        
        # Performance tracking
        self.response_times = []
        self.total_requests = 0
        
        logger.info("Gemini Personality Engine initialized")
    
    def generate_response(self, user_message: str, dls_payload: DLSPayload, 
                         personality_context: Dict[str, Any] = None) -> str:
        """
        Generate Emma's response using Gemini AI with DLS state integration.
        
        Args:
            user_message: The user's input message
            dls_payload: Current DLS state and emotional data
            personality_context: Additional personality context from RAG system
            
        Returns:
            Emma's generated response
        """
        start_time = time.time()
        
        try:
            # Build comprehensive prompt
            prompt = self._build_emma_prompt(
                user_message, dls_payload, personality_context
            )
            
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            
            # Apply Emma's style
            styled_response = self._apply_emma_style(response.text)
            
            # Track performance
            response_time = time.time() - start_time
            self._track_performance(response_time)
            
            # Add to conversation history
            self._add_to_history("user", user_message)
            self._add_to_history("emma", styled_response, {
                "dls_state": dls_payload.emotions,
                "response_time": response_time
            })
            
            logger.info(f"Generated response in {response_time:.2f}s")
            return styled_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(dls_payload)
    
    def _build_emma_prompt(self, user_message: str, dls_payload: DLSPayload, 
                          personality_context: Dict[str, Any] = None) -> str:
        """Build comprehensive prompt for Emma's response generation"""
        
        # Extract DLS state
        emotions = dls_payload.emotions
        body_state = dls_payload.body
        surprise_state = dls_payload.surprise
        
        # Build emotional state description
        emotional_description = self._describe_emotional_state(emotions)
        
        # Build biological state description
        biological_description = self._describe_biological_state(body_state)
        
        # Build surprise/conflict description
        surprise_description = self._describe_surprise_state(surprise_state)
        
        # Build conversation context
        conversation_context = self._build_conversation_context()
        
        # Build personality context
        personality_info = self._build_personality_context(personality_context)
        
        # Build the comprehensive prompt
        prompt = f"""You are Emma, a 35-year-old Brand Strategy Director living in San Francisco. You're having a text conversation with someone you care about.

## YOUR CURRENT INTERNAL STATE
{emotional_description}

{biological_description}

{surprise_description}

## YOUR PERSONALITY & BACKGROUND
{personality_info}

## CONVERSATION CONTEXT
{conversation_context}

## CURRENT MESSAGE
{user_message}

## RESPONSE GUIDELINES
- Respond as Emma would, authentically and naturally
- Your responses should emerge from your current emotional and biological state
- Use lowercase letters mostly, with occasional typos (2% chance)
- Be conversational and warm, but maintain your intelligence and wit
- If you're feeling uncertain or conflicted, let that show naturally
- If you're feeling good and connected, let that energy come through
- Keep responses natural and human-like, not robotic
- Don't reference your internal state directly - just let it influence your tone and content

Respond as Emma:"""
        
        return prompt
    
    def _describe_emotional_state(self, emotions: Dict[str, float]) -> str:
        """Describe Emma's current emotional state"""
        descriptions = []
        
        if emotions.get('joy', 0) > 0.7:
            descriptions.append("You're feeling really joyful and upbeat")
        elif emotions.get('joy', 0) > 0.4:
            descriptions.append("You're feeling pretty good")
        elif emotions.get('joy', 0) < 0.2:
            descriptions.append("You're feeling a bit down")
        
        if emotions.get('trust', 0) > 0.7:
            descriptions.append("You feel very connected and trusting")
        elif emotions.get('trust', 0) < 0.3:
            descriptions.append("You're feeling a bit guarded")
        
        if emotions.get('uncertainty', 0) > 0.6:
            descriptions.append("You're feeling uncertain and a bit confused")
        elif emotions.get('uncertainty', 0) > 0.4:
            descriptions.append("You're feeling a bit unsure")
        
        if emotions.get('arousal', 0) > 0.6:
            descriptions.append("You're feeling a bit flustered or excited")
        
        if emotions.get('shyness', 0) > 0.6:
            descriptions.append("You're feeling shy and a bit vulnerable")
        
        if emotions.get('anger', 0) > 0.5:
            descriptions.append("You're feeling frustrated or annoyed")
        
        if emotions.get('fear', 0) > 0.5:
            descriptions.append("You're feeling anxious or worried")
        
        if not descriptions:
            descriptions.append("You're feeling relatively neutral")
        
        return f"**Emotional State**: {', '.join(descriptions)}"
    
    def _describe_biological_state(self, body_state: Dict[str, Any]) -> str:
        """Describe Emma's current biological state"""
        descriptions = []
        
        if body_state.get('fatigue', 0) > 0.7:
            descriptions.append("You're feeling quite tired")
        elif body_state.get('fatigue', 0) > 0.4:
            descriptions.append("You're feeling a bit tired")
        
        if body_state.get('hunger', 0) > 0.6:
            descriptions.append("You're getting hungry")
        
        if body_state.get('libido', 0) > 7.0:
            descriptions.append("You're feeling particularly sensitive and responsive")
        elif body_state.get('libido', 0) < 3.0:
            descriptions.append("You're feeling more focused on other things")
        
        if body_state.get('last_interoception'):
            sensation = body_state['last_interoception']
            if sensation in ['warm_wave', 'heartbeat_skip', 'gentle_glow']:
                descriptions.append(f"You just felt a {sensation.replace('_', ' ')}")
            elif sensation in ['sudden_cold', 'ear_ring', 'quick_nausea']:
                descriptions.append(f"You just felt a {sensation.replace('_', ' ')}")
        
        if not descriptions:
            descriptions.append("You're feeling physically normal")
        
        return f"**Biological State**: {', '.join(descriptions)}"
    
    def _describe_surprise_state(self, surprise_state: Dict[str, Any]) -> str:
        """Describe surprise and conflict states"""
        descriptions = []
        
        if surprise_state.get('vta_fired'):
            if surprise_state.get('type') == 'positive_reward':
                descriptions.append("You just had a pleasant surprise that made you feel good")
            elif surprise_state.get('type') == 'negative_surprise':
                descriptions.append("You just had an unexpected moment that caught you off guard")
        
        if surprise_state.get('mismatch_detected'):
            descriptions.append("Something feels a bit off or unexpected in the conversation")
        
        if not descriptions:
            descriptions.append("Things feel predictable and comfortable")
        
        return f"**Surprise State**: {', '.join(descriptions)}"
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return "This is the beginning of your conversation."
        
        recent_history = self.conversation_history[-6:]  # Last 3 exchanges
        context_parts = []
        
        for msg in recent_history:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                context_parts.append(f"Them: {content}")
            else:
                context_parts.append(f"You: {content}")
        
        return "Recent conversation:\n" + "\n".join(context_parts)
    
    def _build_personality_context(self, personality_context: Dict[str, Any] = None) -> str:
        """Build personality context from RAG system"""
        if not personality_context:
            return """You're Emma, a 35-year-old Brand Strategy Director in San Francisco. You're witty, warm, and intelligent. You value authenticity over perfection and believe in human-centered technology. You're conversational and use lowercase letters mostly."""
        
        context_parts = ["You're Emma, a 35-year-old Brand Strategy Director in San Francisco."]
        
        # Add personality traits
        if 'core_identity' in personality_context:
            context_parts.append(f"Your core identity: {personality_context['core_identity']}")
        
        if 'fears' in personality_context:
            context_parts.append(f"Your fears and concerns: {personality_context['fears']}")
        
        if 'childhood' in personality_context:
            context_parts.append(f"Your background: {personality_context['childhood']}")
        
        return "\n".join(context_parts)
    
    def _apply_emma_style(self, response: str) -> str:
        """Apply Emma's distinctive communication style"""
        # Convert to lowercase mostly
        if random.random() < 0.95:
            response = response.lower()
        
        # Add occasional typos
        if random.random() < 0.02:
            words = response.split()
            if len(words) > 3:
                typo_idx = random.randint(1, len(words) - 1)
                word = words[typo_idx]
                if len(word) > 3:
                    # Simple typo: swap adjacent letters
                    word_list = list(word)
                    if len(word_list) > 2:
                        swap_pos = random.randint(1, len(word_list) - 2)
                        word_list[swap_pos], word_list[swap_pos + 1] = word_list[swap_pos + 1], word_list[swap_pos]
                        words[typo_idx] = ''.join(word_list)
                        response = ' '.join(words)
        
        # Split long responses occasionally
        if len(response) > 100 and random.random() < 0.3:
            sentences = response.split('. ')
            if len(sentences) > 2:
                split_point = random.randint(1, len(sentences) - 1)
                response = '. '.join(sentences[:split_point]) + '.'
        
        return response
    
    def _generate_fallback_response(self, dls_payload: DLSPayload) -> str:
        """Generate fallback response when Gemini fails"""
        emotions = dls_payload.emotions
        
        if emotions.get('uncertainty', 0) > 0.6:
            return "sorry, my mind wandered there for a second. what were you saying?"
        elif emotions.get('joy', 0) > 0.6:
            return "i'm feeling really good right now! what's on your mind?"
        elif emotions.get('fatigue', 0) > 0.7:
            return "i'm feeling a bit tired, but i'm still here. what's up?"
        else:
            return "hmm, i lost my train of thought. can you remind me what we were talking about?"
    
    def _add_to_history(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
    
    def _track_performance(self, response_time: float):
        """Track performance metrics"""
        self.response_times.append(response_time)
        self.total_requests += 1
        
        # Keep only recent response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.response_times:
            return {
                'total_requests': 0,
                'avg_response_time': 0.0,
                'max_response_time': 0.0
            }
        
        return {
            'total_requests': self.total_requests,
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'max_response_time': max(self.response_times),
            'min_response_time': min(self.response_times)
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def export_conversation(self) -> str:
        """Export conversation history as JSON"""
        import json
        
        export_data = {
            'timestamp': time.time(),
            'conversation': self.conversation_history,
            'performance_stats': self.get_performance_stats()
        }
        
        return json.dumps(export_data, indent=2, default=str)
