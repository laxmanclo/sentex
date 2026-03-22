"""Operation-scoped metrics collection with optional Prometheus export."""
from .collector import MetricsCollector, OperationMetrics, make_prometheus_exporter
