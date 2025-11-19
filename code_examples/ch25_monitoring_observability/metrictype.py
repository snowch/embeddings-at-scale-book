# Code from Chapter 25
# Book: Embeddings at Scale

"""
Comprehensive Performance Monitoring for Embedding Systems

Architecture:
1. Metric collection: Instrument all critical paths with metrics
2. Aggregation: Time-series database (Prometheus) for metrics storage
3. Visualization: Real-time dashboards (Grafana) for monitoring
4. Alerting: Rule-based alerts on metric thresholds and anomalies
5. Tracing: Distributed tracing for request flow analysis

Metrics categories:
- Query metrics: Latency, throughput, error rates, timeout rates
- Index metrics: Build time, memory usage, query accuracy, fragmentation
- Cache metrics: Hit rates, eviction rates, size, staleness
- Resource metrics: CPU, memory, GPU, disk I/O, network
- Business metrics: Cost per query, user satisfaction, downstream impact

Dashboard goals:
- Real-time system health overview
- Quick identification of performance bottlenecks
- Trend analysis for capacity planning
- Drill-down for root cause analysis
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"        # Monotonically increasing (total queries)
    GAUGE = "gauge"           # Point-in-time value (current QPS, memory usage)
    HISTOGRAM = "histogram"   # Distribution of values (latency percentiles)
    SUMMARY = "summary"       # Similar to histogram but client-side aggregation

@dataclass
class MetricValue:
    """Single metric observation"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class PerformanceSnapshot:
    """
    Point-in-time performance snapshot
    
    Captures all key metrics for dashboard display
    """
    timestamp: datetime

    # Query metrics
    queries_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float
    timeout_rate: float
    error_rate: float

    # Index metrics
    index_memory_gb: float
    index_query_accuracy: float
    candidates_scanned_avg: int

    # Cache metrics
    cache_hit_rate: float
    cache_memory_gb: float
    cache_eviction_rate: float

    # Resource metrics
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    disk_iops: float
    network_mbps: float

    # Cost metrics
    cost_per_query_usd: float
    total_cost_hourly_usd: float

    # Quality metrics
    quality_score: float
    drift_score: float

class PerformanceMonitor:
    """
    Real-time performance monitoring system
    
    Collects, aggregates, and exposes metrics for dashboards and alerting.
    """

    def __init__(
        self,
        window_size_seconds: int = 300,  # 5 minute window
        retention_hours: int = 24
    ):
        """
        Initialize performance monitor
        
        Args:
            window_size_seconds: Time window for aggregations
            retention_hours: How long to retain detailed metrics
        """
        self.window_size = timedelta(seconds=window_size_seconds)
        self.retention = timedelta(hours=retention_hours)

        # Thread-safe metric storage
        self.lock = threading.Lock()

        # Recent metric values for aggregation
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Aggregated snapshots
        self.snapshots: deque = deque(maxlen=retention_hours * 12)  # 5-min snapshots

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Start background aggregation
        self.running = True
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric observation"""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=metric_type
        )

        with self.lock:
            self.metrics[name].append(metric)

            # Prune old metrics
            cutoff = datetime.now() - self.retention
            while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                self.metrics[name].popleft()

    def record_query(
        self,
        latency_ms: float,
        success: bool,
        timed_out: bool,
        cache_hit: bool,
        candidates_scanned: int,
        index_name: str = "default"
    ):
        """Convenience method to record query metrics"""
        self.record_metric("query_latency_ms", latency_ms,
                          labels={"index": index_name}, metric_type=MetricType.HISTOGRAM)
        self.record_metric("query_count", 1,
                          labels={"index": index_name, "success": str(success)},
                          metric_type=MetricType.COUNTER)

        if timed_out:
            self.record_metric("query_timeout", 1,
                              labels={"index": index_name}, metric_type=MetricType.COUNTER)

        if not success:
            self.record_metric("query_error", 1,
                              labels={"index": index_name}, metric_type=MetricType.COUNTER)

        if cache_hit:
            self.record_metric("cache_hit", 1, metric_type=MetricType.COUNTER)
        else:
            self.record_metric("cache_miss", 1, metric_type=MetricType.COUNTER)

        self.record_metric("candidates_scanned", candidates_scanned,
                          labels={"index": index_name}, metric_type=MetricType.HISTOGRAM)

    def record_resource_usage(
        self,
        cpu_percent: float,
        memory_gb: float,
        gpu_percent: Optional[float] = None,
        disk_iops: Optional[float] = None,
        network_mbps: Optional[float] = None
    ):
        """Record resource utilization metrics"""
        self.record_metric("cpu_utilization", cpu_percent, metric_type=MetricType.GAUGE)
        self.record_metric("memory_gb", memory_gb, metric_type=MetricType.GAUGE)

        if gpu_percent is not None:
            self.record_metric("gpu_utilization", gpu_percent, metric_type=MetricType.GAUGE)
        if disk_iops is not None:
            self.record_metric("disk_iops", disk_iops, metric_type=MetricType.GAUGE)
        if network_mbps is not None:
            self.record_metric("network_mbps", network_mbps, metric_type=MetricType.GAUGE)

    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot for dashboard"""
        now = datetime.now()
        window_start = now - self.window_size

        with self.lock:
            # Query metrics
            latencies = [m.value for m in self.metrics.get("query_latency_ms", [])
                        if m.timestamp >= window_start]
            query_counts = [m for m in self.metrics.get("query_count", [])
                           if m.timestamp >= window_start]

            # Calculate QPS
            if query_counts:
                total_queries = sum(m.value for m in query_counts)
                time_span = (now - query_counts[0].timestamp).total_seconds()
                qps = total_queries / max(time_span, 1)
            else:
                qps = 0.0

            # Latency percentiles
            if latencies:
                avg_latency = np.mean(latencies)
                p50 = np.percentile(latencies, 50)
                p90 = np.percentile(latencies, 90)
                p99 = np.percentile(latencies, 99)
                p999 = np.percentile(latencies, 99.9) if len(latencies) > 1000 else p99
            else:
                avg_latency = p50 = p90 = p99 = p999 = 0.0

            # Error and timeout rates
            error_counts = sum(m.value for m in self.metrics.get("query_error", [])
                              if m.timestamp >= window_start)
            timeout_counts = sum(m.value for m in self.metrics.get("query_timeout", [])
                                if m.timestamp >= window_start)
            total_queries = sum(m.value for m in query_counts)

            error_rate = error_counts / max(total_queries, 1)
            timeout_rate = timeout_counts / max(total_queries, 1)

            # Cache metrics
            cache_hits = sum(m.value for m in self.metrics.get("cache_hit", [])
                            if m.timestamp >= window_start)
            cache_misses = sum(m.value for m in self.metrics.get("cache_miss", [])
                              if m.timestamp >= window_start)
            cache_hit_rate = cache_hits / max(cache_hits + cache_misses, 1)

            # Get latest gauge values
            def get_latest_gauge(name: str, default: float = 0.0) -> float:
                values = [m.value for m in self.metrics.get(name, [])
                         if m.timestamp >= window_start]
                return values[-1] if values else default

            snapshot = PerformanceSnapshot(
                timestamp=now,
                queries_per_second=qps,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50,
                p90_latency_ms=p90,
                p99_latency_ms=p99,
                p999_latency_ms=p999,
                timeout_rate=timeout_rate,
                error_rate=error_rate,
                index_memory_gb=get_latest_gauge("index_memory_gb"),
                index_query_accuracy=get_latest_gauge("index_query_accuracy", 0.95),
                candidates_scanned_avg=int(np.mean([m.value for m in self.metrics.get("candidates_scanned", [])
                                                     if m.timestamp >= window_start]) or 0),
                cache_hit_rate=cache_hit_rate,
                cache_memory_gb=get_latest_gauge("cache_memory_gb"),
                cache_eviction_rate=get_latest_gauge("cache_eviction_rate"),
                cpu_utilization=get_latest_gauge("cpu_utilization"),
                memory_utilization=get_latest_gauge("memory_gb"),
                gpu_utilization=get_latest_gauge("gpu_utilization"),
                disk_iops=get_latest_gauge("disk_iops"),
                network_mbps=get_latest_gauge("network_mbps"),
                cost_per_query_usd=get_latest_gauge("cost_per_query_usd"),
                total_cost_hourly_usd=get_latest_gauge("total_cost_hourly_usd"),
                quality_score=get_latest_gauge("quality_score", 85.0),
                drift_score=get_latest_gauge("drift_score")
            )

        return snapshot

    def _aggregation_loop(self):
        """Background thread for periodic snapshot aggregation"""
        while self.running:
            try:
                snapshot = self.get_current_snapshot()

                with self.lock:
                    self.snapshots.append(snapshot)

                # Check alerts
                self._check_alerts(snapshot)

            except Exception as e:
                print(f"Error in aggregation loop: {e}")

            time.sleep(60)  # Aggregate every minute

    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check if any alert conditions are met"""
        alerts = []

        # High latency alerts
        if snapshot.p99_latency_ms > 100:
            alerts.append(f"High p99 latency: {snapshot.p99_latency_ms:.1f}ms > 100ms")

        # High error rate
        if snapshot.error_rate > 0.01:  # >1%
            alerts.append(f"High error rate: {snapshot.error_rate*100:.2f}% > 1%")

        # High timeout rate
        if snapshot.timeout_rate > 0.005:  # >0.5%
            alerts.append(f"High timeout rate: {snapshot.timeout_rate*100:.2f}% > 0.5%")

        # Low cache hit rate
        if snapshot.cache_hit_rate < 0.5:  # <50%
            alerts.append(f"Low cache hit rate: {snapshot.cache_hit_rate*100:.1f}% < 50%")

        # High resource utilization
        if snapshot.cpu_utilization > 90:
            alerts.append(f"High CPU utilization: {snapshot.cpu_utilization:.1f}% > 90%")

        if snapshot.memory_utilization > 32:  # >32GB
            alerts.append(f"High memory usage: {snapshot.memory_utilization:.1f}GB > 32GB")

        # Low quality score
        if snapshot.quality_score < 70:
            alerts.append(f"Low quality score: {snapshot.quality_score:.1f} < 70")

        # High drift score
        if snapshot.drift_score > 0.3:
            alerts.append(f"High drift score: {snapshot.drift_score:.3f} > 0.3")

        # Trigger alert callbacks
        if alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(snapshot, alerts)
                except Exception as e:
                    print(f"Error in alert callback: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)

    def get_dashboard_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get data for dashboard display"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self.lock:
            recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff]

        if not recent_snapshots:
            return {"error": "No data available"}

        return {
            "current": self._snapshot_to_dict(recent_snapshots[-1]),
            "history": {
                "timestamps": [s.timestamp.isoformat() for s in recent_snapshots],
                "qps": [s.queries_per_second for s in recent_snapshots],
                "p50_latency": [s.p50_latency_ms for s in recent_snapshots],
                "p99_latency": [s.p99_latency_ms for s in recent_snapshots],
                "error_rate": [s.error_rate * 100 for s in recent_snapshots],
                "cache_hit_rate": [s.cache_hit_rate * 100 for s in recent_snapshots],
                "cpu_utilization": [s.cpu_utilization for s in recent_snapshots],
                "quality_score": [s.quality_score for s in recent_snapshots],
                "drift_score": [s.drift_score for s in recent_snapshots]
            },
            "summary": {
                "avg_qps": np.mean([s.queries_per_second for s in recent_snapshots]),
                "avg_latency": np.mean([s.avg_latency_ms for s in recent_snapshots]),
                "max_latency": max([s.p99_latency_ms for s in recent_snapshots]),
                "avg_error_rate": np.mean([s.error_rate for s in recent_snapshots]),
                "avg_cache_hit_rate": np.mean([s.cache_hit_rate for s in recent_snapshots]),
                "total_cost": sum([s.total_cost_hourly_usd for s in recent_snapshots]) * (hours / len(recent_snapshots))
            }
        }

    def _snapshot_to_dict(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Convert snapshot to dictionary"""
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "qps": snapshot.queries_per_second,
            "latency": {
                "avg": snapshot.avg_latency_ms,
                "p50": snapshot.p50_latency_ms,
                "p90": snapshot.p90_latency_ms,
                "p99": snapshot.p99_latency_ms,
                "p999": snapshot.p999_latency_ms
            },
            "errors": {
                "error_rate": snapshot.error_rate,
                "timeout_rate": snapshot.timeout_rate
            },
            "cache": {
                "hit_rate": snapshot.cache_hit_rate,
                "memory_gb": snapshot.cache_memory_gb,
                "eviction_rate": snapshot.cache_eviction_rate
            },
            "resources": {
                "cpu": snapshot.cpu_utilization,
                "memory_gb": snapshot.memory_utilization,
                "gpu": snapshot.gpu_utilization,
                "disk_iops": snapshot.disk_iops,
                "network_mbps": snapshot.network_mbps
            },
            "costs": {
                "per_query_usd": snapshot.cost_per_query_usd,
                "hourly_usd": snapshot.total_cost_hourly_usd
            },
            "quality": {
                "score": snapshot.quality_score,
                "drift": snapshot.drift_score
            }
        }

    def generate_dashboard_html(self, hours: int = 1) -> str:
        """Generate simple HTML dashboard"""
        data = self.get_dashboard_data(hours)

        if "error" in data:
            return f"<html><body><h1>Error: {data['error']}</h1></body></html>"

        current = data["current"]
        summary = data["summary"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Embedding System Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; margin-bottom: 5px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .metric-unit {{ font-size: 16px; color: #95a5a6; }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        .section {{ background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #ecf0f1; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Embedding System Performance Dashboard</h1>
            <p>Last updated: {current['timestamp']}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Queries Per Second</div>
                <div class="metric-value {'good' if current['qps'] > 100 else 'warning' if current['qps'] > 10 else 'bad'}">
                    {current['qps']:.1f} <span class="metric-unit">QPS</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">P99 Latency</div>
                <div class="metric-value {'good' if current['latency']['p99'] < 50 else 'warning' if current['latency']['p99'] < 100 else 'bad'}">
                    {current['latency']['p99']:.1f} <span class="metric-unit">ms</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Error Rate</div>
                <div class="metric-value {'good' if current['errors']['error_rate'] < 0.01 else 'warning' if current['errors']['error_rate'] < 0.05 else 'bad'}">
                    {current['errors']['error_rate']*100:.2f} <span class="metric-unit">%</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value {'good' if current['cache']['hit_rate'] > 0.7 else 'warning' if current['cache']['hit_rate'] > 0.5 else 'bad'}">
                    {current['cache']['hit_rate']*100:.1f} <span class="metric-unit">%</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Quality Score</div>
                <div class="metric-value {'good' if current['quality']['score'] > 80 else 'warning' if current['quality']['score'] > 70 else 'bad'}">
                    {current['quality']['score']:.1f} <span class="metric-unit">/100</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Hourly Cost</div>
                <div class="metric-value">
                    ${current['costs']['hourly_usd']:.2f} <span class="metric-unit">USD/hr</span>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Latency Distribution</div>
            <table>
                <tr>
                    <th>Percentile</th>
                    <th>Latency (ms)</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>P50</td>
                    <td>{current['latency']['p50']:.1f}</td>
                    <td class="{'good' if current['latency']['p50'] < 20 else 'warning' if current['latency']['p50'] < 50 else 'bad'}">
                        {'✓ Good' if current['latency']['p50'] < 20 else '⚠ OK' if current['latency']['p50'] < 50 else '✗ Slow'}
                    </td>
                </tr>
                <tr>
                    <td>P90</td>
                    <td>{current['latency']['p90']:.1f}</td>
                    <td class="{'good' if current['latency']['p90'] < 50 else 'warning' if current['latency']['p90'] < 100 else 'bad'}">
                        {'✓ Good' if current['latency']['p90'] < 50 else '⚠ OK' if current['latency']['p90'] < 100 else '✗ Slow'}
                    </td>
                </tr>
                <tr>
                    <td>P99</td>
                    <td>{current['latency']['p99']:.1f}</td>
                    <td class="{'good' if current['latency']['p99'] < 100 else 'warning' if current['latency']['p99'] < 200 else 'bad'}">
                        {'✓ Good' if current['latency']['p99'] < 100 else '⚠ OK' if current['latency']['p99'] < 200 else '✗ Slow'}
                    </td>
                </tr>
                <tr>
                    <td>P99.9</td>
                    <td>{current['latency']['p999']:.1f}</td>
                    <td class="{'good' if current['latency']['p999'] < 200 else 'warning' if current['latency']['p999'] < 500 else 'bad'}">
                        {'✓ Good' if current['latency']['p999'] < 200 else '⚠ OK' if current['latency']['p999'] < 500 else '✗ Slow'}
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Resource Utilization</div>
            <table>
                <tr>
                    <th>Resource</th>
                    <th>Current</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>CPU</td>
                    <td>{current['resources']['cpu']:.1f}%</td>
                    <td class="{'good' if current['resources']['cpu'] < 70 else 'warning' if current['resources']['cpu'] < 90 else 'bad'}">
                        {'✓ Normal' if current['resources']['cpu'] < 70 else '⚠ High' if current['resources']['cpu'] < 90 else '✗ Critical'}
                    </td>
                </tr>
                <tr>
                    <td>Memory</td>
                    <td>{current['resources']['memory_gb']:.1f} GB</td>
                    <td class="{'good' if current['resources']['memory_gb'] < 24 else 'warning' if current['resources']['memory_gb'] < 32 else 'bad'}">
                        {'✓ Normal' if current['resources']['memory_gb'] < 24 else '⚠ High' if current['resources']['memory_gb'] < 32 else '✗ Critical'}
                    </td>
                </tr>
                <tr>
                    <td>GPU</td>
                    <td>{current['resources']['gpu']:.1f}%</td>
                    <td class="{'good' if current['resources']['gpu'] < 80 else 'warning' if current['resources']['gpu'] < 95 else 'bad'}">
                        {'✓ Normal' if current['resources']['gpu'] < 80 else '⚠ High' if current['resources']['gpu'] < 95 else '✗ Critical'}
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Summary (Last {hours} hour{'s' if hours != 1 else ''})</div>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average QPS</td>
                    <td>{summary['avg_qps']:.1f}</td>
                </tr>
                <tr>
                    <td>Average Latency</td>
                    <td>{summary['avg_latency']:.1f} ms</td>
                </tr>
                <tr>
                    <td>Max P99 Latency</td>
                    <td>{summary['max_latency']:.1f} ms</td>
                </tr>
                <tr>
                    <td>Average Error Rate</td>
                    <td>{summary['avg_error_rate']*100:.2f}%</td>
                </tr>
                <tr>
                    <td>Average Cache Hit Rate</td>
                    <td>{summary['avg_cache_hit_rate']*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Total Cost</td>
                    <td>${summary['total_cost']:.2f}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
        return html

    def shutdown(self):
        """Shutdown monitoring"""
        self.running = False
        if self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5)


# Example usage
if __name__ == "__main__":
    import random

    # Initialize monitor
    monitor = PerformanceMonitor()

    # Register alert callback
    def alert_handler(snapshot: PerformanceSnapshot, alerts: List[str]):
        print(f"\n🚨 ALERT at {snapshot.timestamp.isoformat()}:")
        for alert in alerts:
            print(f"  - {alert}")

    monitor.register_alert_callback(alert_handler)

    # Simulate queries
    print("Simulating query traffic...")
    for i in range(1000):
        # Simulate varying query patterns
        success = random.random() > 0.02  # 2% error rate
        timed_out = random.random() < 0.005  # 0.5% timeout rate
        cache_hit = random.random() < 0.7  # 70% cache hit rate

        # Latency varies by cache hit
        if cache_hit:
            latency = random.gauss(5, 2)  # Fast cache hit
        else:
            latency = random.gauss(30, 10)  # Slower database query

        latency = max(1, latency)

        monitor.record_query(
            latency_ms=latency,
            success=success,
            timed_out=timed_out,
            cache_hit=cache_hit,
            candidates_scanned=random.randint(100, 10000)
        )

        # Simulate resource usage
        if i % 100 == 0:
            monitor.record_resource_usage(
                cpu_percent=random.gauss(60, 15),
                memory_gb=random.gauss(16, 3),
                gpu_percent=random.gauss(45, 10),
                disk_iops=random.gauss(1000, 200),
                network_mbps=random.gauss(500, 100)
            )

            # Record cost and quality
            monitor.record_metric("cost_per_query_usd", 0.0001 * random.gauss(1, 0.1))
            monitor.record_metric("total_cost_hourly_usd", 2.5 * random.gauss(1, 0.1))
            monitor.record_metric("quality_score", 85 * random.gauss(1, 0.05))
            monitor.record_metric("drift_score", 0.1 * random.gauss(1, 0.5))

        time.sleep(0.01)  # 100 QPS

    # Get current snapshot
    print("\nCurrent Performance Snapshot:")
    snapshot = monitor.get_current_snapshot()
    print(f"  QPS: {snapshot.queries_per_second:.1f}")
    print(f"  P50 latency: {snapshot.p50_latency_ms:.1f}ms")
    print(f"  P99 latency: {snapshot.p99_latency_ms:.1f}ms")
    print(f"  Error rate: {snapshot.error_rate*100:.2f}%")
    print(f"  Cache hit rate: {snapshot.cache_hit_rate*100:.1f}%")
    print(f"  Quality score: {snapshot.quality_score:.1f}")

    # Generate dashboard
    print("\nGenerating dashboard HTML...")
    html = monitor.generate_dashboard_html(hours=1)
    with open("/tmp/dashboard.html", "w") as f:
        f.write(html)
    print("Dashboard saved to /tmp/dashboard.html")

    # Show dashboard data
    print("\nDashboard Data (JSON):")
    dashboard_data = monitor.get_dashboard_data(hours=1)
    print(json.dumps(dashboard_data, indent=2, default=str))

    monitor.shutdown()
