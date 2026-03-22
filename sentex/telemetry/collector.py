"""Operation-scoped telemetry for Sentex.

Collects latency, token counts, confidence scores, and retrieval stats per call.
Optional Prometheus export when prometheus-client is installed.

Usage:
    from sentex.telemetry import MetricsCollector
    from sentex import ContextGraph

    collector = MetricsCollector()
    graph = ContextGraph(metrics=collector)

    # ... run queries ...
    print(collector.summary())
    # → {"ingest": {"count": 3, "mean_ms": 42.1, ...}, "retrieve": {...}}

Prometheus:
    pip install sentex[prometheus]

    from sentex.telemetry import MetricsCollector, make_prometheus_exporter
    collector = MetricsCollector(exporters=[make_prometheus_exporter()])
    # then GET /metrics/prometheus from the server
"""
from __future__ import annotations

import importlib.util
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


# ------------------------------------------------------------------
# Core dataclass
# ------------------------------------------------------------------

@dataclass
class OperationMetrics:
    operation: str          # "ingest" | "retrieve" | "scan_nodes" | "assemble"
    node_id: str | None
    duration_ms: float
    sentences_in: int = 0   # sentences embedded (ingest) or visited (retrieve)
    tokens_out: int = 0     # tokens delivered to caller
    layer_used: str = ""    # "l0" | "l1" | "l2" | "l3"
    confidence: float = 0.0
    converged: bool = False  # BFS converged early (only meaningful for l1)
    cache_hit: bool = False  # embedding was served from cache


# ------------------------------------------------------------------
# Collector
# ------------------------------------------------------------------

class MetricsCollector:
    """Thread-safe rolling buffer of the last N OperationMetrics."""

    def __init__(
        self,
        maxlen: int = 10_000,
        exporters: list | None = None,
    ) -> None:
        self._buf: deque[OperationMetrics] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._exporters: list = exporters or []

    def record(self, op: OperationMetrics) -> None:
        with self._lock:
            self._buf.append(op)
        for exp in self._exporters:
            try:
                exp.export(op)
            except Exception:
                pass

    @contextmanager
    def measure(
        self,
        operation: str,
        node_id: str | None = None,
        **extra,
    ) -> Generator[OperationMetrics, None, None]:
        """Context manager that times a block and records metrics on exit.

        Usage:
            with collector.measure("retrieve", node_id="resources/search") as m:
                result = graph.retrieve(...)
                m.tokens_out = len(result)
                m.confidence = 0.87
        """
        op = OperationMetrics(
            operation=operation,
            node_id=node_id,
            duration_ms=0.0,
            **extra,
        )
        t0 = time.perf_counter()
        try:
            yield op
        finally:
            op.duration_ms = (time.perf_counter() - t0) * 1000
            self.record(op)

    def summary(self) -> dict:
        """Aggregated stats per operation type."""
        with self._lock:
            ops = list(self._buf)
        if not ops:
            return {}

        by_op: dict[str, list[OperationMetrics]] = {}
        for op in ops:
            by_op.setdefault(op.operation, []).append(op)

        result = {}
        for name, entries in by_op.items():
            durations = sorted(e.duration_ms for e in entries)
            n = len(durations)
            result[name] = {
                "count": n,
                "mean_ms": round(sum(durations) / n, 2),
                "p50_ms": round(durations[n // 2], 2),
                "p95_ms": round(durations[max(0, int(n * 0.95) - 1)], 2),
                "p99_ms": round(durations[max(0, int(n * 0.99) - 1)], 2),
                "mean_confidence": round(
                    sum(e.confidence for e in entries) / n, 4
                ),
                "mean_tokens_out": round(
                    sum(e.tokens_out for e in entries) / n, 1
                ),
                "convergence_rate": round(
                    sum(1 for e in entries if e.converged) / n, 3
                ),
                "cache_hit_rate": round(
                    sum(1 for e in entries if e.cache_hit) / n, 3
                ),
            }
        return result

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()

    def recent(self, n: int = 100) -> list[OperationMetrics]:
        with self._lock:
            return list(self._buf)[-n:]


# ------------------------------------------------------------------
# Optional Prometheus exporter
# ------------------------------------------------------------------

def make_prometheus_exporter():
    """Return a PrometheusExporter if prometheus-client is installed, else None."""
    if importlib.util.find_spec("prometheus_client") is None:
        return None

    import prometheus_client as prom  # type: ignore[import]

    class _PrometheusExporter:
        def __init__(self) -> None:
            self._duration = prom.Histogram(
                "sentex_operation_duration_ms",
                "Sentex operation duration in milliseconds",
                ["operation"],
                buckets=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
            )
            self._confidence = prom.Histogram(
                "sentex_retrieval_confidence",
                "Retrieval confidence score",
                ["operation"],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
            self._tokens = prom.Counter(
                "sentex_tokens_total",
                "Total tokens delivered to callers",
                ["operation"],
            )
            self._operations = prom.Counter(
                "sentex_operations_total",
                "Total operations by type",
                ["operation"],
            )

        def export(self, op: OperationMetrics) -> None:
            self._duration.labels(operation=op.operation).observe(op.duration_ms)
            self._operations.labels(operation=op.operation).inc()
            if op.confidence > 0:
                self._confidence.labels(operation=op.operation).observe(op.confidence)
            if op.tokens_out > 0:
                self._tokens.labels(operation=op.operation).inc(op.tokens_out)

        def text_output(self) -> str:
            return prom.generate_latest().decode()

    return _PrometheusExporter()
