"""Bottom-up failure-mode clustering and taxonomy optimization.

General over any audit dataset of (id, model, outcome_class, rationale) records.
See ``optimize`` for the end-to-end loop and ``backends`` for model plumbing.
"""
from .backends import LLMBackend, MockBackend, OpenAICompatBackend
from .fitness import Evaluator, Fitness, adjusted_rand_index, mi_stats
from .optimize import OptResult, optimize
from .schema import Clustering, Mode, Verdict, load_verdicts

__all__ = [
    "Verdict", "Mode", "Clustering", "load_verdicts",
    "LLMBackend", "MockBackend", "OpenAICompatBackend",
    "Evaluator", "Fitness", "mi_stats", "adjusted_rand_index",
    "optimize", "OptResult",
]
