"""Backward-compatibility shim — imports from sentex.telemetry.collector."""
from .telemetry.collector import *  # noqa: F401, F403
from .telemetry.collector import MetricsCollector, OperationMetrics, make_prometheus_exporter
