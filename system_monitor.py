#!/usr/bin/env python3
"""
System Monitor for Emma's Digital Limbic System
Comprehensive performance monitoring and health dashboard
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    state_stability: float = 1.0
    error_rate: float = 0.0
    session_count: int = 0
    memory_count: int = 0
    api_success_rate: float = 1.0

@dataclass
class AlertThresholds:
    """Alert thresholds for monitoring"""
    max_response_time: float = 5.0
    max_memory_mb: float = 1000.0
    max_cpu_percent: float = 80.0
    min_stability: float = 0.6
    max_error_rate: float = 0.1
    max_sessions: int = 50
    min_api_success_rate: float = 0.8

class SystemMonitor:
    """Comprehensive system monitoring with real-time dashboards"""
    
    def __init__(self, history_size: int = 1000, alert_thresholds: AlertThresholds = None):
        self.history_size = history_size
        self.thresholds = alert_thresholds or AlertThresholds()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.alerts = []
        self.max_alerts = 100
        
        # Component monitors
        self.dls_monitor = None
        self.api_monitor = None
        self.session_monitor = None
        self.memory_monitor = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 10.0  # seconds
        
        # Performance tracking
        self.performance_stats = {
            'system_start_time': time.time(),
            'total_requests': 0,
            'total_errors': 0,
            'peak_memory_mb': 0.0,
            'peak_cpu_percent': 0.0,
            'longest_response_time': 0.0,
            'stability_incidents': 0
        }
        
        logger.info(f"SystemMonitor initialized with {history_size} metric history")
    
    def register_component_monitors(self, dls=None, api=None, session=None, memory=None):
        """Register component monitors for integrated health checking"""
        self.dls_monitor = dls
        self.api_monitor = api
        self.session_monitor = session
        self.memory_monitor = memory
        logger.info("Component monitors registered")
    
    def log_request_metrics(self, response_time: float, memory_usage_mb: float = None, 
                          state_stability: float = 1.0, error_occurred: bool = False):
        """Log metrics for a single request"""
        self.performance_stats['total_requests'] += 1
        
        if error_occurred:
            self.performance_stats['total_errors'] += 1
        
        # Update peaks
        if response_time > self.performance_stats['longest_response_time']:
            self.performance_stats['longest_response_time'] = response_time
        
        if memory_usage_mb and memory_usage_mb > self.performance_stats['peak_memory_mb']:
            self.performance_stats['peak_memory_mb'] = memory_usage_mb
        
        if state_stability < 0.8:
            self.performance_stats['stability_incidents'] += 1
        
        # Create metrics snapshot
        current_metrics = self._collect_current_metrics(
            response_time, memory_usage_mb, state_stability, error_occurred
        )
        
        # Store in history
        self.metrics_history.append(current_metrics)
        
        # Check for alerts
        self._check_alerts(current_metrics)
    
    def _collect_current_metrics(self, response_time: float = 0.0, 
                                memory_usage_mb: float = None, 
                                state_stability: float = 1.0,
                                error_occurred: bool = False) -> PerformanceMetrics:
        """Collect comprehensive current metrics"""
        # System metrics
        try:
            process = psutil.Process()
            system_memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except:
            system_memory_mb = 0.0
            cpu_percent = 0.0
        
        # Update CPU peak
        if cpu_percent > self.performance_stats['peak_cpu_percent']:
            self.performance_stats['peak_cpu_percent'] = cpu_percent
        
        # Component metrics
        session_count = 0
        memory_count = 0
        api_success_rate = 1.0
        
        if self.session_monitor:
            try:
                session_stats = self.session_monitor.get_system_stats()
                session_count = session_stats.get('active_sessions', 0)
            except:
                pass
        
        if self.memory_monitor:
            try:
                memory_stats = self.memory_monitor.get_memory_health_metrics([])
                memory_count = memory_stats.get('total_memories', 0)
            except:
                pass
        
        if self.api_monitor:
            try:
                api_health = self.api_monitor.get_health_status()
                api_success_rate = api_health.get('success_rate', 1.0)
            except:
                pass
        
        # Calculate error rate
        error_rate = 0.0
        if self.performance_stats['total_requests'] > 0:
            error_rate = self.performance_stats['total_errors'] / self.performance_stats['total_requests']
        
        return PerformanceMetrics(
            response_time=response_time,
            memory_usage_mb=memory_usage_mb or system_memory_mb,
            cpu_usage_percent=cpu_percent,
            state_stability=state_stability,
            error_rate=error_rate,
            session_count=session_count,
            memory_count=memory_count,
            api_success_rate=api_success_rate
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        if metrics.response_time > self.thresholds.max_response_time:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"High response time: {metrics.response_time:.2f}s",
                'timestamp': metrics.timestamp,
                'value': metrics.response_time,
                'threshold': self.thresholds.max_response_time
            })
        
        if metrics.memory_usage_mb > self.thresholds.max_memory_mb:
            alerts.append({
                'type': 'memory',
                'severity': 'critical' if metrics.memory_usage_mb > self.thresholds.max_memory_mb * 1.5 else 'warning',
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                'timestamp': metrics.timestamp,
                'value': metrics.memory_usage_mb,
                'threshold': self.thresholds.max_memory_mb
            })
        
        if metrics.cpu_usage_percent > self.thresholds.max_cpu_percent:
            alerts.append({
                'type': 'cpu',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                'timestamp': metrics.timestamp,
                'value': metrics.cpu_usage_percent,
                'threshold': self.thresholds.max_cpu_percent
            })
        
        if metrics.state_stability < self.thresholds.min_stability:
            alerts.append({
                'type': 'stability',
                'severity': 'critical',
                'message': f"Low state stability: {metrics.state_stability:.2f}",
                'timestamp': metrics.timestamp,
                'value': metrics.state_stability,
                'threshold': self.thresholds.min_stability
            })
        
        if metrics.error_rate > self.thresholds.max_error_rate:
            alerts.append({
                'type': 'errors',
                'severity': 'warning',
                'message': f"High error rate: {metrics.error_rate:.1%}",
                'timestamp': metrics.timestamp,
                'value': metrics.error_rate,
                'threshold': self.thresholds.max_error_rate
            })
        
        if metrics.session_count > self.thresholds.max_sessions:
            alerts.append({
                'type': 'sessions',
                'severity': 'warning',
                'message': f"High session count: {metrics.session_count}",
                'timestamp': metrics.timestamp,
                'value': metrics.session_count,
                'threshold': self.thresholds.max_sessions
            })
        
        if metrics.api_success_rate < self.thresholds.min_api_success_rate:
            alerts.append({
                'type': 'api',
                'severity': 'critical',
                'message': f"Low API success rate: {metrics.api_success_rate:.1%}",
                'timestamp': metrics.timestamp,
                'value': metrics.api_success_rate,
                'threshold': self.thresholds.min_api_success_rate
            })
        
        # Add new alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
        
        # Trim alerts to max size
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        if not self.metrics_history:
            return {'status': 'no_data', 'uptime_hours': 0}
        
        # Calculate metrics over recent history
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
        
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_stability = sum(m.state_stability for m in recent_metrics) / len(recent_metrics)
        current_error_rate = recent_metrics[-1].error_rate if recent_metrics else 0.0
        current_api_success = recent_metrics[-1].api_success_rate if recent_metrics else 1.0
        
        # Determine overall health
        health_score = 1.0
        health_issues = []
        
        if avg_response_time > self.thresholds.max_response_time:
            health_score -= 0.2
            health_issues.append("slow_response")
        
        if avg_memory > self.thresholds.max_memory_mb:
            health_score -= 0.3
            health_issues.append("high_memory")
        
        if avg_stability < self.thresholds.min_stability:
            health_score -= 0.4
            health_issues.append("unstable")
        
        if current_error_rate > self.thresholds.max_error_rate:
            health_score -= 0.3
            health_issues.append("high_errors")
        
        if current_api_success < self.thresholds.min_api_success_rate:
            health_score -= 0.2
            health_issues.append("api_issues")
        
        # Classify health status
        if health_score >= 0.9:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.5:
            status = "degraded"
        elif health_score >= 0.3:
            status = "poor"
        else:
            status = "critical"
        
        uptime_hours = (time.time() - self.performance_stats['system_start_time']) / 3600
        
        return {
            'status': status,
            'health_score': health_score,
            'uptime_hours': uptime_hours,
            'issues': health_issues,
            'metrics': {
                'avg_response_time': avg_response_time,
                'avg_memory_mb': avg_memory,
                'avg_cpu_percent': avg_cpu,
                'avg_stability': avg_stability,
                'error_rate': current_error_rate,
                'api_success_rate': current_api_success
            },
            'peaks': {
                'memory_mb': self.performance_stats['peak_memory_mb'],
                'cpu_percent': self.performance_stats['peak_cpu_percent'],
                'response_time': self.performance_stats['longest_response_time']
            },
            'total_requests': self.performance_stats['total_requests'],
            'total_errors': self.performance_stats['total_errors'],
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600])  # Last hour
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        health = self.get_system_health()
        
        # Recent metrics for charts
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 points
        
        chart_data = {
            'timestamps': [m.timestamp for m in recent_metrics],
            'response_times': [m.response_time for m in recent_metrics],
            'memory_usage': [m.memory_usage_mb for m in recent_metrics],
            'cpu_usage': [m.cpu_usage_percent for m in recent_metrics],
            'stability': [m.state_stability for m in recent_metrics],
            'error_rates': [m.error_rate for m in recent_metrics]
        }
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 3600]
        
        return {
            'health': health,
            'charts': chart_data,
            'alerts': recent_alerts[-10:],  # Last 10 alerts
            'component_status': self._get_component_status(),
            'recommendations': self._get_performance_recommendations(health)
        }
    
    def _get_component_status(self) -> Dict[str, Any]:
        """Get status of individual components"""
        status = {}
        
        if self.dls_monitor:
            try:
                # This would be implemented by DLS
                status['dls'] = {'status': 'operational', 'details': 'DLS running normally'}
            except:
                status['dls'] = {'status': 'unknown', 'details': 'Cannot check DLS status'}
        
        if self.api_monitor:
            try:
                api_health = self.api_monitor.get_health_status()
                status['api'] = {
                    'status': api_health['health'],
                    'success_rate': api_health['success_rate'],
                    'details': f"Success rate: {api_health['success_rate']:.1%}"
                }
            except:
                status['api'] = {'status': 'unknown', 'details': 'Cannot check API status'}
        
        if self.session_monitor:
            try:
                session_stats = self.session_monitor.get_system_stats()
                status['sessions'] = {
                    'status': 'healthy' if session_stats['active_sessions'] < 20 else 'busy',
                    'active_count': session_stats['active_sessions'],
                    'details': f"{session_stats['active_sessions']} active sessions"
                }
            except:
                status['sessions'] = {'status': 'unknown', 'details': 'Cannot check session status'}
        
        if self.memory_monitor:
            try:
                memory_stats = self.memory_monitor.get_memory_health_metrics([])
                status['memory'] = {
                    'status': memory_stats['status'],
                    'utilization': memory_stats.get('capacity_utilization', 0.0),
                    'details': f"{memory_stats['total_memories']} memories stored"
                }
            except:
                status['memory'] = {'status': 'unknown', 'details': 'Cannot check memory status'}
        
        return status
    
    def _get_performance_recommendations(self, health: Dict[str, Any]) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if 'slow_response' in health['issues']:
            recommendations.append("Consider optimizing DLS processing or adding response caching")
        
        if 'high_memory' in health['issues']:
            recommendations.append("Run memory consolidation or increase pruning frequency")
        
        if 'unstable' in health['issues']:
            recommendations.append("Check state validation settings and DLS parameter tuning")
        
        if 'high_errors' in health['issues']:
            recommendations.append("Investigate error sources and improve error handling")
        
        if 'api_issues' in health['issues']:
            recommendations.append("Check Gemini API health and rate limiting settings")
        
        # General recommendations based on metrics
        metrics = health['metrics']
        if metrics['avg_memory_mb'] > 500:
            recommendations.append("Consider implementing more aggressive memory management")
        
        if metrics['avg_response_time'] > 3.0:
            recommendations.append("Optimize processing pipeline for faster responses")
        
        if not recommendations:
            recommendations.append("System performing well - no optimizations needed")
        
        return recommendations
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_worker():
            while self.monitoring_active:
                try:
                    # Collect and store current metrics
                    current_metrics = self._collect_current_metrics()
                    self.metrics_history.append(current_metrics)
                    
                    # Check for alerts
                    self._check_alerts(current_metrics)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring worker: {e}")
                
                time.sleep(self.monitor_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def export_metrics(self, filepath: str = None) -> str:
        """Export metrics history to JSON file"""
        if not filepath:
            timestamp = int(time.time())
            filepath = f"emma_metrics_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_time': time.time(),
                'uptime_hours': (time.time() - self.performance_stats['system_start_time']) / 3600,
                'total_metrics': len(self.metrics_history)
            },
            'performance_stats': self.performance_stats,
            'health_summary': self.get_system_health(),
            'metrics_history': [
                {
                    'timestamp': m.timestamp,
                    'response_time': m.response_time,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'state_stability': m.state_stability,
                    'error_rate': m.error_rate,
                    'session_count': m.session_count,
                    'api_success_rate': m.api_success_rate
                }
                for m in self.metrics_history
            ],
            'recent_alerts': self.alerts[-50:]  # Last 50 alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
        return filepath

def test_system_monitor():
    """Test system monitor functionality"""
    print("Testing System Monitor")
    print("=" * 50)
    
    monitor = SystemMonitor(history_size=10)
    
    # Simulate some requests
    import random
    for i in range(15):
        monitor.log_request_metrics(
            response_time=random.uniform(0.5, 4.0),
            memory_usage_mb=random.uniform(100, 800),
            state_stability=random.uniform(0.4, 1.0),
            error_occurred=random.random() < 0.1
        )
        time.sleep(0.1)
    
    # Get health status
    health = monitor.get_system_health()
    print(f"System health: {health['status']}")
    print(f"Health score: {health['health_score']:.2f}")
    print(f"Issues: {health['issues']}")
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"Recent alerts: {len(dashboard['alerts'])}")
    print(f"Recommendations: {dashboard['recommendations'][:2]}")
    
    # Export metrics
    filepath = monitor.export_metrics()
    print(f"Metrics exported to: {filepath}")
    
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == "__main__":
    test_system_monitor()
