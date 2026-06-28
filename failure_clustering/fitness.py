"""The scalar objective the optimizer maximises.

Primary signal is **model-discriminativeness** — `NMI(mode; model)`, the fraction
of "which model produced this" recoverable from the mode. Left alone, NMI is
gameable by over-splitting, so it is paired with an **MDL** description-length
penalty (every extra mode must pay for itself) and a **coherence** reward (a mode
must describe one mechanism). **Leverage** up-weights verifier-fixable FP/FN
clusters. Everything except coherence is pure arithmetic on the (mode × model)
table — cheap and deterministic; coherence is the only backend call and is
cached by member set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import Clustering

# Verifier-fixable outcomes are worth more: an FP/FN cluster you can fix in the
# harness beats a diffuse bucket of genuine agent failures.
OUTCOME_WEIGHT = {"TP": 0.0, "TN": 1.0, "FP": 3.0, "FN": 3.0}

DEFAULT_WEIGHTS = {
    "nmi": 1.0,        # model-discriminativeness (the payoff)
    "quality": 1.0,    # cluster compactness/separation — has an interior optimum
                       # (collapse AND over-splitting both score badly), so it is
                       # the guard the one-sided MDL term could not be.
    "modes": 0.15,     # mild parsimony tie-breaker
    "leverage": 0.1,   # up-weight verifier-fixable FP/FN (constant under split/merge)
}


def _log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


@dataclass
class MIStats:
    mi: float
    h_mode: float
    h_model: float
    nmi_model: float
    nmi_sym: float
    cramers_v: float
    g: float
    chi2: float
    dof: int
    per_mode_mi: dict
    per_mode_kl: dict


def mi_stats(clustering: Clustering) -> MIStats:
    codes, models, N = clustering.count_table()
    T = sum(sum(N[c].values()) for c in codes) or 1
    p_mode = {c: sum(N[c].values()) / T for c in codes}
    p_model = {m: sum(N[c][m] for c in codes) / T for m in models}

    mi = 0.0
    per_kl, per_mi = {}, {}
    for c in codes:
        row = sum(N[c].values()) or 1
        kl = 0.0
        for m in models:
            o = N[c][m]
            if o:
                kl += (o / row) * _log2((o / row) / p_model[m]) if p_model[m] else 0.0
        per_kl[c] = kl
        per_mi[c] = p_mode[c] * kl
        mi += per_mi[c]

    h_mode = -sum(p_mode[c] * _log2(p_mode[c]) for c in codes)
    h_model = -sum(p_model[m] * _log2(p_model[m]) for m in models)
    nmi_model = mi / h_model if h_model else 0.0
    nmi_sym = (2 * mi / (h_mode + h_model)) if (h_mode + h_model) else 0.0

    chi2 = g = 0.0
    for c in codes:
        for m in models:
            o = N[c][m]
            e = p_mode[c] * p_model[m] * T
            if e > 0:
                chi2 += (o - e) ** 2 / e
            if o > 0 and e > 0:
                g += 2 * o * math.log(o / e)
    R, C = len(codes), len(models)
    dof = max((R - 1) * (C - 1), 0)
    cramers_v = math.sqrt(chi2 / (T * (min(R, C) - 1))) if T and min(R, C) > 1 else 0.0
    return MIStats(mi, h_mode, h_model, nmi_model, nmi_sym, cramers_v, g, chi2, dof, per_mi, per_kl)


def mdl_norm(clustering: Clustering) -> float:
    """Normalised description length in [0, ~1]: label-coding cost per rollout
    plus a per-mode overhead. Rises with more (and more granular) modes."""
    codes, models, N = clustering.count_table()
    T = sum(sum(N[c].values()) for c in codes) or 1
    p_mode = {c: sum(N[c].values()) / T for c in codes}
    label_bits = sum(-p_mode[c] * _log2(p_mode[c]) for c in codes)  # = H(mode)
    overhead_bits = len(codes) * _log2(T + 1) / T  # amortised cost to *define* each mode
    return (label_bits + overhead_bits) / _log2(T + 1)


def leverage_norm(clustering: Clustering) -> float:
    codes, models, N = clustering.count_table()
    T = sum(sum(N[c].values()) for c in codes) or 1
    lev = sum(sum(N[c].values()) * OUTCOME_WEIGHT.get(clustering.outcome_of(c), 1.0) for c in codes)
    max_lev = T * max(OUTCOME_WEIGHT.values())
    return lev / max_lev if max_lev else 0.0


def mean_silhouette(clustering: Clustering, geom) -> float:
    """Mean silhouette in embedding space, computed *within* each outcome
    stratum. Collapsing a stratum to one mode scores 0; over-splitting a tight
    blob lowers it; well-separated, compact modes score high. This is the
    descriptive-quality / reconstruction term — deterministic and LLM-free."""
    from collections import defaultdict
    by_oc: dict[str, list[str]] = defaultdict(list)
    for c in clustering.present_codes():
        by_oc[clustering.outcome_of(c)].append(c)
    sils: list[float] = []
    for codes in by_oc.values():
        if len(codes) < 2:
            sils.extend(0.0 for c in codes for _ in clustering.members(c))
            continue
        cent = {c: geom.centroid([v.id for v in clustering.members(c)]) for c in codes}
        for c in codes:
            for v in clustering.members(c):
                x = geom.X[geom.row[v.id]]
                a = 1.0 - x @ cent[c]
                b = min(1.0 - x @ cent[o] for o in codes if o != c)
                sils.append((b - a) / (max(a, b) or 1.0))
    return sum(sils) / len(sils) if sils else 0.0


def adjusted_rand_index(a: list, b: list) -> float:
    """ARI between two labelings of the same items (for stability tests)."""
    from collections import Counter
    n = len(a)
    if n == 0:
        return 1.0
    pair = Counter(zip(a, b))
    ai = Counter(a)
    bi = Counter(b)
    comb2 = lambda x: x * (x - 1) / 2
    index = sum(comb2(v) for v in pair.values())
    exp_a = sum(comb2(v) for v in ai.values())
    exp_b = sum(comb2(v) for v in bi.values())
    total = comb2(n)
    expected = exp_a * exp_b / total if total else 0.0
    maxi = (exp_a + exp_b) / 2
    denom = maxi - expected
    return (index - expected) / denom if denom else 1.0


@dataclass
class Fitness:
    scalar: float
    nmi_model: float
    quality: float
    leverage: float
    n_modes: int
    detail: dict


class Evaluator:
    """Scalar objective. With a :class:`Geometry`, the quality term is silhouette
    (deterministic, LLM-free) so the whole search loop runs without model calls.
    Without geometry it falls back to a cached LLM-judged coherence."""

    def __init__(self, backend, weights: dict | None = None, *, geom=None, use_coherence: bool = True):
        import threading
        self.backend = backend
        self.geom = geom
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.use_coherence = use_coherence
        self._coh_cache: dict[frozenset, float] = {}
        self._lock = threading.Lock()

    def _coherence_of(self, members) -> float:
        """Cached, thread-safe LLM coherence for one member set (fallback path)."""
        key = frozenset(v.id for v in members)
        with self._lock:
            hit = self._coh_cache.get(key)
        if hit is not None:
            return hit
        val = self.backend.coherence([v.text for v in members])  # idempotent; outside lock
        with self._lock:
            self._coh_cache[key] = val
        return val

    def _quality(self, clustering: Clustering) -> float:
        if self.geom is not None:
            return mean_silhouette(clustering, self.geom)
        if not self.use_coherence:
            return 0.0
        codes = clustering.present_codes()
        total = sum(len(clustering.members(c)) for c in codes) or 1
        return sum(self._coherence_of(clustering.members(c)) * len(clustering.members(c))
                   for c in codes) / total

    def fitness(self, clustering: Clustering) -> Fitness:
        mi = mi_stats(clustering)
        quality = self._quality(clustering)
        lev = leverage_norm(clustering)
        n_modes = len(clustering.present_codes())
        n = len(clustering.verdicts) or 1
        scalar = (self.w["nmi"] * mi.nmi_model
                  + self.w["quality"] * quality
                  - self.w["modes"] * (n_modes / n)
                  + self.w["leverage"] * lev)
        return Fitness(
            scalar=scalar, nmi_model=mi.nmi_model, quality=quality,
            leverage=lev, n_modes=n_modes,
            detail={"mi_bits": mi.mi, "h_model": mi.h_model, "cramers_v": mi.cramers_v,
                    "mdl": mdl_norm(clustering), "g": mi.g, "dof": mi.dof,
                    "per_mode_mi": mi.per_mode_mi},
        )

    def scalar(self, clustering: Clustering) -> float:
        return self.fitness(clustering).scalar
