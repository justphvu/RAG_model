import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Represents performance metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_threads: int
    custom_metrics: Dict = None

class PerformanceMonitor:
    """
    Monitors system performance and provides real-time metrics.
    
    Features:
    - System resource monitoring (CPU, Memory, Disk)
    - Custom metric tracking
    - Historical data storage
    - Performance alerts
    """
    
    def __init__(
        self,
        max_history_size: int = 1000,
        monitoring_interval: float = 5.0,
        enable_monitoring: bool = True
    ):
        self.max_history_size = max_history_size
        self.monitoring_interval = monitoring_interval
        self.enable_monitoring = enable_monitoring
        
        # Historical data storage
        self.metrics_history = deque(maxlen=max_history_size)
        
        # Custom metrics
        self.custom_metrics = {}
        
        # Performance thresholds for alerts
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_thread is not None:
            return
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if self.monitoring_thread is None:
            return
        
        self.stop_event.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance alerts
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Active threads
        active_threads = threading.active_count()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            active_threads=active_threads,
            custom_metrics=self.custom_metrics.copy()
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance threshold violations."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > self.thresholds["disk_usage_percent"]:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        for alert in alerts:
            logger.warning(f"Performance alert: {alert}")
    
    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric to track."""
        self.custom_metrics[name] = value
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics."""
        return self._collect_metrics()
    
    def get_metrics_history(self, duration_minutes: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get historical metrics, optionally filtered by duration."""
        if duration_minutes is None:
            return list(self.metrics_history)
        
        cutoff_time = time.time() - (duration_minutes * 60)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_average_metrics(self, duration_minutes: int = 5) -> Dict:
        """Get average metrics over a specified duration."""
        recent_metrics = self.get_metrics_history(duration_minutes)
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_metrics = {
            "cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "memory_used_mb": sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
            "memory_available_mb": sum(m.memory_available_mb for m in recent_metrics) / len(recent_metrics),
            "disk_usage_percent": sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            "active_threads": sum(m.active_threads for m in recent_metrics) / len(recent_metrics),
        }
        
        return avg_metrics
    
    def get_performance_summary(self) -> Dict:
        """Get a comprehensive performance summary."""
        current_metrics = self.get_current_metrics()
        avg_metrics_5min = self.get_average_metrics(5)
        avg_metrics_15min = self.get_average_metrics(15)
        
        return {
            "current": asdict(current_metrics),
            "average_5min": avg_metrics_5min,
            "average_15min": avg_metrics_15min,
            "history_size": len(self.metrics_history),
            "thresholds": self.thresholds,
            "custom_metrics": self.custom_metrics
        }
    
    def set_threshold(self, metric: str, threshold: float) -> None:
        """Set a performance threshold for alerts."""
        if metric in self.thresholds:
            self.thresholds[metric] = threshold
            logger.info(f"Updated threshold for {metric}: {threshold}")
    
    def clear_history(self) -> None:
        """Clear historical metrics."""
        self.metrics_history.clear()
        logger.info("Performance history cleared")

# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()

def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None 