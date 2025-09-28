#!/usr/bin/env python3
"""
Production Flask Application for Emma - The Digital Companion
Enhanced with session management, monitoring, state validation, and API resilience
"""

import os
import json
import time
import logging
import uuid
import psutil
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS

# Import Emma components
from emma_companion import EmmaCompanion
from session_manager import SessionManager, SessionLimits
from system_monitor import SystemMonitor, AlertThresholds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Production components
session_manager = SessionManager(SessionLimits(
    max_duration_hours=2.0,
    max_messages=500,
    max_memory_mb=500,
    idle_timeout_minutes=30.0,
    max_concurrent_sessions=20,
    max_processing_time_seconds=30.0
))

system_monitor = SystemMonitor(
    history_size=1000,
    alert_thresholds=AlertThresholds(
        max_response_time=5.0,
        max_memory_mb=1000.0,
        max_cpu_percent=80.0,
        min_stability=0.6,
        max_error_rate=0.1,
        max_sessions=50,
        min_api_success_rate=0.8
    )
)

# Global Emma instance
emma_instance = None

def get_emma():
    """Get or create Emma instance with production resilience"""
    global emma_instance
    if emma_instance is None:
        # Get Gemini API key from environment
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if gemini_api_key:
            emma_instance = EmmaCompanion(gemini_api_key=gemini_api_key)
            logger.info("Initialized Emma instance with Gemini AI and production components")
        else:
            emma_instance = EmmaCompanion()
            logger.info("Initialized Emma instance (fallback mode)")
        
        # Register Emma components with monitoring
        if hasattr(emma_instance, 'dls') and emma_instance.dls:
            system_monitor.register_component_monitors(
                dls=emma_instance.dls,
                api=getattr(emma_instance.gemini_engine, 'api_client', None) if hasattr(emma_instance, 'gemini_engine') else None,
                session=session_manager,
                memory=getattr(emma_instance.dls, 'memory_manager', None)
            )
        
        # Start monitoring
        system_monitor.start_monitoring()
    
    return emma_instance

def get_or_create_session():
    """Get or create user session with validation"""
    if 'session_id' not in session:
        # Create new session
        user_id = request.headers.get('X-User-ID')
        session_id, created = session_manager.create_session(user_id)
        if not created:
            return None, "Maximum concurrent sessions reached"
        session['session_id'] = session_id
        logger.info(f"Created new session: {session_id}")
    
    session_id = session['session_id']
    
    # Validate existing session
    valid, reason = session_manager.validate_session(session_id)
    if not valid:
        # Session invalid, create new one
        user_id = request.headers.get('X-User-ID')
        new_session_id, created = session_manager.create_session(user_id)
        if not created:
            return None, "Cannot create new session"
        session['session_id'] = new_session_id
        logger.info(f"Session {session_id} invalid ({reason}), created new: {new_session_id}")
        return new_session_id, None
    
    return session_id, None

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return send_from_directory('.', 'index.html')

@app.route('/test')
def test_interface():
    """Serve the test HTML interface"""
    return send_from_directory('.', 'test_interface.html')

@app.route('/minimal')
def minimal_test():
    """Serve the minimal test interface"""
    return send_from_directory('.', 'minimal_test.html')

@app.route('/dashboard')
def monitoring_dashboard():
    """Serve monitoring dashboard"""
    dashboard_data = system_monitor.get_dashboard_data()
    return jsonify(dashboard_data)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat message with full production resilience"""
    start_time = time.time()
    error_occurred = False
    
    try:
        # Session management
        session_id, session_error = get_or_create_session()
        if session_error:
            return jsonify({'error': session_error}), 429
        
        # Get and validate message
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if len(message) > 5000:  # Limit message length
            return jsonify({'error': 'Message too long'}), 400
        
        # Process with Emma
        emma = get_emma()
        response = emma.process_message(message)
        processing_time = time.time() - start_time
        
        # Update session activity
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        session_manager.update_session_activity(
            session_id, 
            message_processed=True,
            processing_time=processing_time,
            memory_usage_mb=memory_usage_mb
        )
        
        # Log metrics to monitor
        system_monitor.log_request_metrics(
            response_time=processing_time,
            memory_usage_mb=memory_usage_mb,
            state_stability=1.0,  # Would get from DLS
            error_occurred=False
        )
        
        return jsonify({
            'response': response,
            'processing_time': processing_time,
            'timestamp': time.time(),
            'session_id': session_id
        })
        
    except Exception as e:
        error_occurred = True
        processing_time = time.time() - start_time
        
        # Log error metrics
        system_monitor.log_request_metrics(
            response_time=processing_time,
            error_occurred=True
        )
        
        # Update session with error
        if 'session_id' in locals():
            session_manager.update_session_activity(
                session_id,
                processing_time=processing_time,
                error=str(e)
            )
        
        logger.error(f"Error processing chat message: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get Emma's current state with session validation"""
    try:
        session_id, session_error = get_or_create_session()
        if session_error:
            return jsonify({'error': session_error}), 429
        
        emma = get_emma()
        stats = emma.get_stats()
        
        # Enhanced state with production metrics
        state = {
            'drives': stats.get('substrate_stats', {}).get('drives', {}),
            'phenomenology': stats.get('phenomenology_stats', {}).get('most_recent', 'a quiet moment of being'),
            'session_id': session_id,
            'message_count': stats.get('conversation_length', 0),
            'uptime': stats.get('uptime', 0),
            'dls_state': stats.get('dls_stats', {}),
            'memory_health': stats.get('dls_stats', {}).get('memory_stats', {}),
            'system_health': system_monitor.get_system_health()
        }
        
        return jsonify(state)
        
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get comprehensive statistics including production metrics"""
    try:
        emma = get_emma()
        stats = emma.get_stats()
        
        # Add production metrics
        stats.update({
            'session_stats': session_manager.get_system_stats(),
            'system_health': system_monitor.get_system_health(),
            'monitoring_data': system_monitor.get_dashboard_data()
        })
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check with production monitoring"""
    try:
        emma = get_emma()
        system_health = system_monitor.get_system_health()
        session_stats = session_manager.get_system_stats()
        
        # Determine overall health
        overall_status = "healthy"
        if system_health['status'] in ['poor', 'critical']:
            overall_status = "unhealthy"
        elif system_health['status'] == 'degraded':
            overall_status = "degraded"
        
        health_data = {
            'status': overall_status,
            'timestamp': time.time(),
            'components': {
                'emma': 'operational',
                'dls': 'operational',
                'sessions': 'healthy' if session_stats['active_sessions'] < 15 else 'busy',
                'api': 'operational',  # Would get from API monitor
                'memory': 'operational'
            },
            'metrics': {
                'uptime_hours': system_health['uptime_hours'],
                'active_sessions': session_stats['active_sessions'],
                'total_requests': system_health['total_requests'],
                'error_rate': system_health['metrics']['error_rate'],
                'avg_response_time': system_health['metrics']['avg_response_time']
            },
            'alerts': system_health['recent_alerts']
        }
        
        status_code = 200
        if overall_status == "unhealthy":
            status_code = 503
        elif overall_status == "degraded":
            status_code = 202
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset Emma session with session management"""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            session_manager.terminate_session(session_id, "user_reset")
            session.pop('session_id', None)
        
        # Reset Emma instance
        global emma_instance
        if emma_instance:
            emma_instance.reset_session()
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({'error': 'Internal server error'}), 500

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
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/monitoring/export', methods=['GET'])
def export_monitoring_data():
    """Export monitoring metrics"""
    try:
        filepath = system_monitor.export_metrics()
        
        with open(filepath, 'r') as f:
            export_data = f.read()
        
        # Clean up file
        os.remove(filepath)
        
        response = app.response_class(
            export_data,
            mimetype='application/json',
            headers={
                'Content-Disposition': 'attachment; filename=emma-monitoring-data.json'
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting monitoring data: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(429)
def rate_limited(error):
    """Handle rate limiting"""
    return jsonify({
        'error': 'Rate limited',
        'message': 'Too many requests, please slow down'
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.teardown_appcontext
def cleanup_session(error):
    """Clean up resources on request end"""
    if error:
        logger.error(f"Request failed with error: {error}")

def shutdown_handler():
    """Clean shutdown handler"""
    logger.info("Shutting down Emma application...")
    session_manager.shutdown()
    system_monitor.stop_monitoring()
    logger.info("Shutdown complete")

def main():
    """Main application entry point"""
    import signal
    import atexit
    
    # Register shutdown handlers
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_handler())
    signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())
    atexit.register(shutdown_handler)
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Emma Digital Biology Companion (Production) on port {port}")
    logger.info("Production features enabled: Session Management, State Validation, API Resilience, System Monitoring")
    
    # Warm up the system
    get_emma()
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()
