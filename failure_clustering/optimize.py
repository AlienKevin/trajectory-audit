"""End-to-end taxonomy optimization.

verdicts → mechanism summaries → consensus clustering → critic-driven
split/merge hill-climb → round-trip fidelity + boundary triage. Returns the
optimized clustering plus a report comparing it to the initial one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collections import defaultdict

from .cluster import Geometry, build_clustering, cosine_dist, relabel_all
from .fitness import Evaluator, mi_stats
from .parallel import DEFAULT_WORKERS
from .schema import OUTCOME_CLASSES, Clustering, Verdict
from .search import critic_loop

DEFAULT_LENSES = ["failure mechanism", "root cause", "fix type"]

# Default per-stratum mode budget for non-focus outcome classes. The focus
# stratum (e.g. TN) absorbs whatever is left of ``max_modes``.
_NONFOCUS_DEFAULT = {"TP": 1, "FP": 1, "FN": 3, "TN": 4}


def allocate_caps(by_outcome, max_modes, focus, base_caps=None):
    """Split a global mode budget across outcome strata.

    Each populated stratum gets a per-mode cap; ``focus`` (e.g. "TN") receives
    the remainder so the budget concentrates where you want detail. Explicit
    ``base_caps`` (e.g. ``{"TP": 1}``) always win. Result sums to ≤ ``max_modes``.
    """
    base_caps = dict(base_caps or {})
    populated = [oc for oc in OUTCOME_CLASSES if by_outcome.get(oc)]
    caps: dict[str, int] = {}
    for oc in populated:
        if oc == focus:
            continue
        size = len(by_outcome[oc])
        caps[oc] = max(1, min(base_caps.get(oc, _NONFOCUS_DEFAULT.get(oc, 3)), size))
    if focus in populated:
        remaining = max_modes - sum(caps.values())
        caps[focus] = max(1, min(len(by_outcome[focus]), base_caps.get(focus, remaining), remaining))
    # trim non-focus strata if we still overflow (focus is protected last)
    while sum(caps.values()) > max_modes:
        trimmable = [o for o in caps if o != focus and caps[o] > 1]
        target = max(trimmable, key=lambda o: caps[o]) if trimmable else focus
        caps[target] = max(1, caps[target] - 1)
    return caps


@dataclass
class OptResult:
    clustering: Clustering
    initial: Clustering
    history: list[dict]
    report: dict
    triage: list[dict] = field(default_factory=list)


def _with_mechanism(verdicts: list[Verdict], backend) -> list[Verdict]:
    mechs = backend.mechanism([v.text for v in verdicts])
    out = []
    for v, m in zip(verdicts, mechs):
        out.append(Verdict(id=v.id, model=v.model, outcome_class=v.outcome_class,
                           text=m or v.text, task=v.task, raw=v.raw))
    return out


def _silhouette_triage(clustering: Clustering, geom: Geometry, top: int = 15) -> list[dict]:
    """Boundary cases: lowest silhouette within their outcome stratum — the only
    rollouts worth a human glance."""
    from collections import defaultdict
    by_oc: dict[str, list[str]] = defaultdict(list)
    code_of = clustering.assignment
    for c in clustering.present_codes():
        by_oc[clustering.outcome_of(c)].append(c)
    rows = []
    for oc, codes in by_oc.items():
        cent = {c: geom.centroid([v.id for v in clustering.members(c)]) for c in codes}
        for c in codes:
            for v in clustering.members(c):
                x = geom.X[geom.row[v.id]]
                a = 1.0 - x @ cent[c]
                others = [1.0 - x @ cent[o] for o in codes if o != c]
                b = min(others) if others else a
                sil = (b - a) / (max(a, b) or 1.0)
                rows.append({"rollout_id": v.id, "model": v.model, "outcome_class": oc,
                             "mode": code_of[v.id], "silhouette": round(sil, 3)})
    rows.sort(key=lambda r: r["silhouette"])
    return rows[:top]


def _fidelity(clustering: Clustering, backend, *, sample_negatives: int = 8) -> dict:
    """Round-trip: give the backend only a mode's definition and ask it to
    predict membership. Low recall ⇒ a loose/uninformative definition."""
    scores = {}
    all_ids = list(clustering.assignment)
    for c in clustering.present_codes():
        members = clustering.members(c)
        defn = clustering.modes[c].definition or clustering.modes[c].name
        pred_pos = backend.predict_membership(defn, [v.text for v in members])
        recall = sum(pred_pos) / len(members) if members else 0.0
        neg = [clustering.verdict(i) for i in all_ids if clustering.assignment[i] != c][:sample_negatives]
        pred_neg = backend.predict_membership(defn, [v.text for v in neg]) if neg else []
        fpr = sum(pred_neg) / len(neg) if neg else 0.0
        scores[c] = {"recall": round(recall, 2), "false_positive_rate": round(fpr, 2)}
    return scores


def optimize(
    verdicts: list[Verdict],
    backend,
    *,
    weights: dict | None = None,
    lenses: list[str] | None = DEFAULT_LENSES,
    k_per_outcome: dict[str, int] | None = None,
    use_mechanism: bool = True,
    use_coherence: bool = True,
    fidelity: bool = True,
    max_iters: int = 30,
    critic_rounds: int = 2,
    workers: int = DEFAULT_WORKERS,
    max_modes: int | None = None,
    focus_outcome: str | None = None,
    outcome_caps: dict[str, int] | None = None,
) -> OptResult:
    work = _with_mechanism(verdicts, backend) if use_mechanism else verdicts

    # Resolve the mode budget. With max_modes set, build the initial clustering
    # *at* the caps (consensus k = cap per stratum) so the budget is spent where
    # we want it — otherwise the cheap mechanism model may never find a
    # fitness-improving split of a big homogeneous stratum.
    caps = None
    if max_modes is not None:
        by_outcome = defaultdict(list)
        for v in work:
            by_outcome[v.outcome_class].append(v)
        caps = outcome_caps if outcome_caps and not focus_outcome else \
            allocate_caps(by_outcome, max_modes, focus_outcome, outcome_caps)
        k_per_outcome = {**caps, **(k_per_outcome or {})}

    geom = Geometry.build(work, backend)
    ev = Evaluator(backend, weights, geom=geom, use_coherence=use_coherence)

    initial = build_clustering(work, backend, k_per_outcome=k_per_outcome, lenses=lenses, workers=workers)
    fit0 = ev.fitness(initial)

    final, history = critic_loop(initial, geom, ev, rounds=critic_rounds, max_iters=max_iters,
                                 workers=workers, caps=caps, max_modes=max_modes)
    final = relabel_all(final, backend, workers=workers)  # name the final taxonomy
    fit1 = ev.fitness(final)

    triage = _silhouette_triage(final, geom)
    report = {
        "backend": getattr(backend, "name", "?"),
        "n_rollouts": len(verdicts),
        "max_modes": max_modes,
        "caps": caps,
        "initial": _fit_dict(fit0, initial),
        "final": _fit_dict(fit1, final),
        "delta_nmi_model": round(fit1.nmi_model - fit0.nmi_model, 4),
        "delta_scalar": round(fit1.scalar - fit0.scalar, 4),
        "history": history,
    }
    if fidelity:
        report["fidelity"] = _fidelity(final, backend)
    return OptResult(clustering=final, initial=initial, history=history, report=report, triage=triage)


def _fit_dict(fit, clustering: Clustering) -> dict:
    mi = mi_stats(clustering)
    return {
        "scalar": round(fit.scalar, 4),
        "nmi_model": round(fit.nmi_model, 4),
        "mi_bits": round(mi.mi, 4),
        "cramers_v": round(mi.cramers_v, 4),
        "quality": round(fit.quality, 4),
        "leverage": round(fit.leverage, 4),
        "n_modes": fit.n_modes,
    }
