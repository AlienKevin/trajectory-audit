"""Tiny thread-pool helper for fanning out I/O-bound backend calls.

LLM/embedding calls are network-bound, so threads give near-linear speedup while
the GIL is released during the HTTP wait. ``pmap`` preserves input order (so
selection/tie-breaking stays deterministic) and falls back to a plain loop for
``workers <= 1`` or trivially small inputs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

DEFAULT_WORKERS = 8


def pmap(fn, items, workers: int = DEFAULT_WORKERS):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))  # ordered; re-raises first exception
