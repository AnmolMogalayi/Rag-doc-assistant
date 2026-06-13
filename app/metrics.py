"""Minimal in-process metrics (no external dependency).

Exposes counters + latency summaries at GET /metrics in a Prometheus-style text format.
Thread-safe; good enough for a single-node deployment and easy to scrape or eyeball.
For multi-node, replace with prometheus_client and a shared registry.
"""
from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[str, float] = defaultdict(float)
_latency_sum: dict[str, float] = defaultdict(float)
_latency_count: dict[str, int] = defaultdict(int)


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def observe_latency(name: str, ms: float, **labels: str) -> None:
    key = _key(name, labels)
    with _lock:
        _latency_sum[key] += ms
        _latency_count[key] += 1


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


def render() -> str:
    lines = ["# RAG doc assistant metrics"]
    with _lock:
        for key, val in sorted(_counters.items()):
            lines.append(f"{key} {val}")
        for key, total in sorted(_latency_sum.items()):
            count = _latency_count[key] or 1
            lines.append(f"{key}_latency_ms_sum {total}")
            lines.append(f"{key}_latency_ms_count {_latency_count[key]}")
            lines.append(f"{key}_latency_ms_avg {round(total / count, 2)}")
    return "\n".join(lines) + "\n"


def snapshot() -> dict:
    with _lock:
        out = {"counters": dict(_counters), "latency_avg_ms": {}}
        for key, total in _latency_sum.items():
            count = _latency_count[key] or 1
            out["latency_avg_ms"][key] = round(total / count, 2)
        return out
