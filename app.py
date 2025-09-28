#!/usr/bin/env python3
"""
Flask web application for Emma Digital Biology Companion
Provides REST API for the web interface
"""

import os
import json
import time
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Emma components
from emma_companion import EmmaCompanion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global Emma instance
emma_instance = None

def get_emma():
    """Get or create Emma instance"""
    global emma_instance
    if emma_instance is None:
        emma_instance = EmmaCompanion()
        logger.info("Initialized Emma instance")
    return emma_instance

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat message"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'error': 'Message is required'
            }), 400
        
        emma = get_emma()
        start_time = time.time()
        
        response = emma.process_message(message)
        processing_time = time.time() - start_time
        
        return jsonify({
            'response': response,
            'processing_time': processing_time,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get Emma's current state"""
    try:
        emma = get_emma()
        stats = emma.get_stats()
        
        # Format state for frontend
        state = {
            'drives': stats['substrate_stats']['drives'],
            'phenomenology': stats['phenomenology_stats'].get('most_recent', 'a quiet moment of being'),
            'session_id': stats['session_id'],
            'message_count': stats['conversation_length'],
            'uptime': stats.get('uptime', 0)
        }
        
        return jsonify(state)
        
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get comprehensive statistics"""
    try:
        emma = get_emma()
        stats = emma.get_stats()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset Emma session"""
    try:
        global emma_instance
        
        if emma_instance:
            emma_instance.reset_session()
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/export', methods=['GET'])
def export_conversation():
    """Export conversation history"""
    try:
        emma = get_emma()
        export_data = emma.export_conversation()
        
        response = app.response_class(
            export_data,
            mimetype='application/json',
            headers={
                'Content-Disposition': 'attachment; filename=emma-conversation.json'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        emma = get_emma()
        stats = emma.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'session_id': stats['session_id'],
            'uptime': stats.get('uptime', 0)
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error'
    }), 500

def main():
    """Main application entry point"""
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Emma Digital Biology Companion on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()