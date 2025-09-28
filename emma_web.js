class EmmaWebInterface {
    constructor() {
        this.emma = null;
        this.isProcessing = false;
        this.startTime = Date.now();
        this.messageCount = 0;
        
        this.initializeElements();
        this.initializeEmma();
        this.bindEvents();
        this.startStatsUpdate();
    }

    initializeElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // State elements
        this.stateElements = {
            fatigue: {
                value: document.getElementById('fatigueValue'),
                bar: document.getElementById('fatigueBar')
            },
            curiosity: {
                value: document.getElementById('curiosityValue'),
                bar: document.getElementById('curiosityBar')
            },
            social: {
                value: document.getElementById('socialValue'),
                bar: document.getElementById('socialBar')
            },
            stability: {
                value: document.getElementById('stabilityValue'),
                bar: document.getElementById('stabilityBar')
            }
        };
        
        this.phenomenologyText = document.getElementById('phenomenologyText');
        
        // Stats elements
        this.statsElements = {
            messageCount: document.getElementById('messageCount'),
            responseTime: document.getElementById('responseTime'),
            memoryUsage: document.getElementById('memoryUsage'),
            uptime: document.getElementById('uptime')
        };
        
        // Control buttons
        this.resetBtn = document.getElementById('resetBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        // Navigation
        this.navBtns = document.querySelectorAll('.nav-btn');
    }

    async initializeEmma() {
        try {
            // Show loading state
            this.showLoadingMessage('Initializing Emma...');
            
            // Initialize Emma (this would normally load from the Python backend)
            // For now, we'll simulate the initialization
            await this.simulateEmmaInitialization();
            
            this.hideLoadingMessage();
            this.addMessage('emma', 'Hi there! I\'m Emma. I\'m here to chat, share thoughts, and maybe help you see things from a different perspective. How are you doing today?');
            
        } catch (error) {
            console.error('Error initializing Emma:', error);
            this.hideLoadingMessage();
            this.addMessage('emma', 'I\'m having a bit of trouble getting started. Mind giving me a moment?');
        }
    }

    async simulateEmmaInitialization() {
        // Simulate initialization delay
        return new Promise(resolve => setTimeout(resolve, 1000));
    }

    bindEvents() {
        // Send message events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Control button events
        this.resetBtn.addEventListener('click', () => this.resetSession());
        this.exportBtn.addEventListener('click', () => this.exportChat());
        this.clearBtn.addEventListener('click', () => this.clearMemory());

        // Navigation events
        this.navBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Auto-resize input
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        });
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;

        // Clear input and add user message
        this.chatInput.value = '';
        this.addMessage('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Process message
        this.isProcessing = true;
        this.setLoadingState(true);
        
        try {
            const startTime = Date.now();
            
            // In a real implementation, this would call the Python backend
            const response = await this.processMessageWithEmma(message);
            
            const responseTime = Date.now() - startTime;
            
            this.hideTypingIndicator();
            this.addMessage('emma', response);
            
            // Update stats
            this.messageCount++;
            this.updateStats(responseTime);
            
            // Update Emma's state display
            this.updateEmmaState();
            
        } catch (error) {
            console.error('Error processing message:', error);
            this.hideTypingIndicator();
            this.addMessage('emma', 'Sorry, I\'m having a moment. Can you try that again?');
        } finally {
            this.isProcessing = false;
            this.setLoadingState(false);
        }
    }

    async processMessageWithEmma(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data.response;
        } catch (error) {
            console.error('Error calling backend:', error);
            // Fallback to simulated response if backend fails
            return this.generateResponse(message);
        }
    }

    generateResponse(message) {
        const messageLower = message.toLowerCase();
        
        // Simple response patterns based on message content
        if (messageLower.includes('how are') || messageLower.includes('how\'s')) {
            return this.generateHowAreYouResponse();
        } else if (messageLower.includes('what') && messageLower.includes('you')) {
            return this.generateAboutMeResponse();
        } else if (messageLower.includes('work') || messageLower.includes('job')) {
            return this.generateWorkResponse();
        } else if (messageLower.includes('feel') || messageLower.includes('feeling')) {
            return this.generateFeelingResponse();
        } else {
            return this.generateGeneralResponse();
        }
    }

    generateHowAreYouResponse() {
        const responses = [
            "i\'m doing well, thanks for asking. been thinking about authenticity a lot lately - how about you?",
            "honestly? a little tired but good. working on this project that\'s both exciting and exhausting",
            "i\'m here and present, which feels like a win these days. what\'s going on with you?",
            "doing okay. had a long day of meetings but juniper\'s been keeping me company. how are you holding up?"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    generateAboutMeResponse() {
        const responses = [
            "i work in brand strategy, helping tech companies sound more human. it\'s weirdly fulfilling work",
            "i\'m 35, live in sf with my dog juniper, and spend way too much time thinking about how stories shape reality",
            "i\'m someone who believes authenticity beats perfection every time, though i don\'t always live up to that",
            "i\'m emma - i ask too many questions and care too much about whether technology is helping or hurting human connection"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    generateWorkResponse() {
        const responses = [
            "work\'s been intense lately. i\'m helping this startup figure out how to talk about their ai without sounding like every other ai company",
            "it\'s funny - i spend my days teaching companies how to be authentic, but sometimes i wonder if i\'m being authentic enough in my own life",
            "brand strategy is weird work. it\'s part psychology, part storytelling, part trying to convince people that being real is better than being perfect",
            "i had this meeting today where someone actually asked if we could make their brand more \"relatably authentic\" and i wanted to scream"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    generateFeelingResponse() {
        const responses = [
            "i\'ve been thinking about how feelings are like weather - they pass through but they\'re not who you are",
            "honestly? a mix of hopeful and overwhelmed. there\'s so much happening in the world and i want to make a difference but sometimes feel so small",
            "i feel curious today. like there\'s something important i\'m supposed to be learning but i haven\'t figured out what it is yet",
            "i\'m feeling that particular kind of tired that comes from caring too much about too many things"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    generateGeneralResponse() {
        const responses = [
            "that\'s interesting. tell me more about what you\'re thinking",
            "i hadn\'t thought about it that way before. what made you see it like that?",
            "there\'s something about what you\'re saying that feels important. can you say more?",
            "i\'m really listening. what else is on your mind?",
            "you know what\'s funny? that makes me think about something completely different but somehow connected",
            "i love how conversations can go in these unexpected directions. where do you think this is taking us?"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'emma' ? 'E' : 'Y';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Animate in
        setTimeout(() => {
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'E';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            dotsDiv.appendChild(dot);
        }
        
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(dotsDiv);
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    showLoadingMessage(message) {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message emma loading';
        loadingDiv.id = 'loadingMessage';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'E';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;
        
        loadingDiv.appendChild(avatar);
        loadingDiv.appendChild(content);
        
        this.chatMessages.appendChild(loadingDiv);
        this.scrollToBottom();
    }

    hideLoadingMessage() {
        const loadingMessage = document.getElementById('loadingMessage');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    setLoadingState(loading) {
        if (loading) {
            this.chatInput.disabled = true;
            this.sendBtn.disabled = true;
            this.chatInput.classList.add('loading');
        } else {
            this.chatInput.disabled = false;
            this.sendBtn.disabled = false;
            this.chatInput.classList.remove('loading');
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    async updateEmmaState() {
        try {
            const response = await fetch('/api/state');
            if (response.ok) {
                const data = await response.json();
                
                // Update state bars with real data
                const states = {
                    fatigue: data.drives?.fatigue || 0.0,
                    curiosity: data.drives?.curiosity || 0.7,
                    social: data.drives?.social || 0.5,
                    stability: data.drives?.stability || 0.9
                };
                
                Object.keys(states).forEach(key => {
                    const value = states[key];
                    const elements = this.stateElements[key];
                    
                    if (elements) {
                        elements.value.textContent = value.toFixed(2);
                        elements.bar.style.width = `${value * 100}%`;
                    }
                });
                
                // Update phenomenology with real data
                if (data.phenomenology) {
                    this.phenomenologyText.textContent = data.phenomenology;
                }
                
                // Update stats with real data
                if (data.message_count !== undefined) {
                    this.statsElements.messageCount.textContent = data.message_count;
                }
                
            } else {
                // Fallback to simulated state if API fails
                this.updateEmmaStateSimulated();
            }
        } catch (error) {
            console.error('Error fetching Emma state:', error);
            this.updateEmmaStateSimulated();
        }
    }
    
    updateEmmaStateSimulated() {
        // Simulate Emma's state changes (fallback)
        const states = {
            fatigue: Math.min(1.0, 0.1 + Math.random() * 0.3),
            curiosity: Math.max(0.1, 0.7 + (Math.random() - 0.5) * 0.4),
            social: Math.max(0.1, 0.5 + (Math.random() - 0.5) * 0.3),
            stability: Math.max(0.5, 0.9 + (Math.random() - 0.5) * 0.2)
        };
        
        Object.keys(states).forEach(key => {
            const value = states[key];
            const elements = this.stateElements[key];
            
            if (elements) {
                elements.value.textContent = value.toFixed(2);
                elements.bar.style.width = `${value * 100}%`;
            }
        });
        
        // Update phenomenology
        const phenomenologies = [
            "a pleasant lightness in my chest",
            "thoughts moving with unusual clarity",
            "a subtle sharpening of sounds and colors",
            "breath finding a slower rhythm",
            "a gentle warmth spreading through my limbs",
            "a slight forward tilt in posture",
            "a pleasant tension across the shoulders",
            "time compressing into focused moments"
        ];
        
        if (Math.random() < 0.3) { // 30% chance to update
            this.phenomenologyText.textContent = phenomenologies[Math.floor(Math.random() * phenomenologies.length)];
        }
    }

    updateStats(responseTime) {
        this.statsElements.messageCount.textContent = this.messageCount;
        this.statsElements.responseTime.textContent = `${(responseTime / 1000).toFixed(1)}s`;
        
        // Simulate memory usage
        const memoryMB = (2.1 + Math.random() * 0.5).toFixed(1);
        this.statsElements.memoryUsage.textContent = `${memoryMB}MB`;
        
        // Update uptime
        const uptime = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(uptime / 60);
        const seconds = uptime % 60;
        this.statsElements.uptime.textContent = `${minutes}m ${seconds}s`;
    }

    startStatsUpdate() {
        // Update stats every 5 seconds
        setInterval(() => {
            this.updateStats(0);
            
            // Occasionally update Emma's state
            if (Math.random() < 0.1) { // 10% chance every 5 seconds
                this.updateEmmaState();
            }
        }, 5000);
    }

    async resetSession() {
        if (confirm('Are you sure you want to reset the session? This will clear all conversation history.')) {
            this.setLoadingState(true);
            
            try {
                // Call backend reset API
                const response = await fetch('/api/reset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    // Reset local state
                    this.messageCount = 0;
                    this.startTime = Date.now();
                    
                    // Clear chat
                    this.chatMessages.innerHTML = '';
                    
                    // Reset stats
                    this.updateStats(0);
                    await this.updateEmmaState();
                    
                    // Add welcome message
                    this.addMessage('emma', 'Hi there! I\'m Emma. I\'m here to chat, share thoughts, and maybe help you see things from a different perspective. How are you doing today?');
                } else {
                    throw new Error('Failed to reset session on backend');
                }
                
            } catch (error) {
                console.error('Error resetting session:', error);
                this.addMessage('emma', 'Sorry, I had trouble resetting. Let me try a different approach...');
            } finally {
                this.setLoadingState(false);
            }
        }
    }

    exportChat() {
        const chatData = {
            timestamp: new Date().toISOString(),
            messages: Array.from(this.chatMessages.children)
                .filter(el => el.classList.contains('message'))
                .map(el => ({
                    role: el.classList.contains('user') ? 'user' : 'emma',
                    content: el.querySelector('.message-content').textContent,
                    timestamp: new Date().toISOString()
                }))
        };
        
        const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `emma-chat-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    clearMemory() {
        if (confirm('Are you sure you want to clear Emma\'s memory? This cannot be undone.')) {
            // Simulate memory clearing
            this.showLoadingMessage('Clearing memory...');
            
            setTimeout(() => {
                this.hideLoadingMessage();
                this.addMessage('emma', 'Memory cleared. I feel... lighter, somehow. Like I\'ve had a good night\'s sleep.');
            }, 2000);
        }
    }

    switchTab(tabName) {
        // Update navigation
        this.navBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // In a real implementation, this would switch between different views
        console.log(`Switching to tab: ${tabName}`);
    }
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new EmmaWebInterface();
});

// Add some utility functions for debugging
window.emmaDebug = {
    addMessage: (role, content) => {
        const event = new CustomEvent('addMessage', { detail: { role, content } });
        document.dispatchEvent(event);
    },
    
    updateState: (state) => {
        const event = new CustomEvent('updateState', { detail: state });
        document.dispatchEvent(event);
    }
};