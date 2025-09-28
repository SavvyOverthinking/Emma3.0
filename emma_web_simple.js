// Simple Emma Web Interface - Debug Version
class EmmaWebInterface {
    constructor() {
        console.log("EmmaWebInterface constructor called");
        this.isProcessing = false;
        this.messageCount = 0;
        this.startTime = Date.now();
        
        this.initializeElements();
        this.bindEvents();
        this.addMessage('emma', 'Hi there! I\'m Emma. I\'m here to chat, share thoughts, and maybe help you see things from a different perspective. How are you doing today?');
    }

    initializeElements() {
        console.log("Initializing elements...");
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        console.log("Elements found:", {
            chatMessages: !!this.chatMessages,
            chatInput: !!this.chatInput,
            sendBtn: !!this.sendBtn
        });
    }

    bindEvents() {
        console.log("Binding events...");
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => {
                console.log("Send button clicked");
                this.sendMessage();
            });
        }
        
        if (this.chatInput) {
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    console.log("Enter key pressed");
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
    }

    async sendMessage() {
        console.log("sendMessage called");
        const message = this.chatInput ? this.chatInput.value.trim() : '';
        console.log("Message:", message);
        
        if (!message || this.isProcessing) {
            console.log("Message empty or processing:", { message, isProcessing: this.isProcessing });
            return;
        }

        console.log("Processing message...");
        this.isProcessing = true;
        
        // Clear input and add user message
        if (this.chatInput) {
            this.chatInput.value = '';
        }
        this.addMessage('user', message);
        
        try {
            console.log("Calling API...");
            const response = await this.callAPI(message);
            console.log("API response:", response);
            this.addMessage('emma', response);
            this.messageCount++;
        } catch (error) {
            console.error('Error processing message:', error);
            this.addMessage('emma', 'Sorry, I\'m having a moment. Can you try that again?');
        } finally {
            this.isProcessing = false;
        }
    }

    async callAPI(message) {
        console.log("Calling API with message:", message);
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        console.log("API response status:", response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("API response data:", data);
        return data.response;
    }

    addMessage(role, content) {
        console.log("Adding message:", { role, content });
        if (!this.chatMessages) {
            console.error("chatMessages element not found");
            return;
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.textContent = content;
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded, initializing Emma...");
    new EmmaWebInterface();
});
