"""Bottom-up clustering: rationales → a labelled, stratified taxonomy.

Average-linkage agglomerative clustering (UPGMA, numpy-only) run *within each
outcome class*, then each cluster is named by the backend with a falsifiable
membership test. ``consensus_cluster`` runs several embedding "lenses" and
fuses them through a co-assignment matrix — the single biggest lever against
embedding/LLM nondeterminism.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .features import cosine_dist
from .parallel import DEFAULT_WORKERS, pmap
from .schema import Clustering, Mode, Verdict


def agglomerative(D: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage (UPGMA) clustering. Returns labels in ``[0, k)``."""
    n = D.shape[0]
    k = max(1, min(k, n))
    if n == 0:
        return np.zeros(0, dtype=int)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    dist = D.astype(float).copy()
    np.fill_diagonal(dist, np.inf)
    active = list(range(n))

    while len(active) > k:
        # find closest pair among active clusters
        best = (np.inf, -1, -1)
        for ai in range(len(active)):
            i = active[ai]
            for aj in range(ai + 1, len(active)):
                j = active[aj]
                d = dist[i, j]
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        # merge j into i (Lance-Williams average-linkage update)
        ni, nj = sizes[i], sizes[j]
        for m in active:
            if m in (i, j):
                continue
            dist[i, m] = dist[m, i] = (ni * dist[i, m] + nj * dist[j, m]) / (ni + nj)
        members[i].extend(members[j])
        sizes[i] = ni + nj
        active.remove(j)

    labels = np.empty(n, dtype=int)
    for lab, c in enumerate(active):
        for idx in members[c]:
            labels[idx] = lab
    return labels


def suggest_k(n: int) -> int:
    """A modest starting granularity; the optimizer refines it via split/merge."""
    if n <= 3:
        return 1
    return max(2, min(8, round(math.sqrt(n / 2))))


def _embed_lens(backend, texts: list[str], lens: str | None) -> np.ndarray:
    prompt = texts if lens is None else [f"[{lens}] {t}" for t in texts]
    return backend.embed(prompt)


def _coassignment(label_sets: list[np.ndarray]) -> np.ndarray:
    """Fraction of lenses in which each pair lands in the same cluster."""
    n = len(label_sets[0])
    C = np.zeros((n, n))
    for labels in label_sets:
        same = labels[:, None] == labels[None, :]
        C += same
    C /= len(label_sets)
    return C


def cluster_stratum(
    verdicts: list[Verdict],
    backend,
    *,
    outcome_class: str,
    k: int | None = None,
    lenses: list[str] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[int, list[Verdict]]:
    """Cluster one outcome stratum; return {local_label: [verdicts]}."""
    if not verdicts:
        return {}
    texts = [v.text for v in verdicts]
    k = k or suggest_k(len(verdicts))
    if lenses and len(lenses) > 1 and len(verdicts) > k:
        dmats = pmap(lambda ln: cosine_dist(_embed_lens(backend, texts, ln)), lenses, workers)
        label_sets = [agglomerative(D, k) for D in dmats]
        C = _coassignment(label_sets)
        labels = agglomerative(1.0 - C, k)  # consensus over co-assignment
    else:
        lens = (lenses or [None])[0]
        labels = agglomerative(cosine_dist(_embed_lens(backend, texts, lens)), k)
    out: dict[int, list[Verdict]] = defaultdict(list)
    for v, lab in zip(verdicts, labels):
        out[int(lab)].append(v)
    return out


def build_clustering(
    verdicts: list[Verdict],
    backend,
    *,
    k_per_outcome: dict[str, int] | None = None,
    lenses: list[str] | None = None,
    label: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> Clustering:
    """Initial stratified, labelled clustering over all outcome classes."""
    by_outcome: dict[str, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_outcome[v.outcome_class].append(v)

    cluster_list: list[tuple[str, list[Verdict]]] = []
    for oc, members in by_outcome.items():
        k = (k_per_outcome or {}).get(oc)
        clusters = cluster_stratum(members, backend, outcome_class=oc, k=k, lenses=lenses, workers=workers)
        cluster_list.extend((oc, vs) for vs in clusters.values())

    # Label every cluster concurrently (independent API calls), then register
    # sequentially so code de-duplication stays deterministic.
    if label:
        infos = pmap(lambda cv: backend.label([v.text for v in cv[1]], cv[0]), cluster_list, workers)
    else:
        infos = [None] * len(cluster_list)

    assignment: dict[str, str] = {}
    modes: dict[str, Mode] = {}
    for (oc, vs), info in zip(cluster_list, infos):
        code = _register_cluster(info, vs, oc, modes)
        for v in vs:
            assignment[v.id] = code
    return Clustering(verdicts=verdicts, assignment=assignment, modes=modes)


def _register_cluster(info: dict | None, vs: list[Verdict], oc: str, modes: dict[str, Mode]) -> str:
    """Create a Mode from a (possibly precomputed) label. Mutates ``modes``;
    call sequentially."""
    if info is not None:
        code = _dedup(f"{oc}_{info['code']}", modes)
        modes[code] = Mode(
            code=code, name=info["name"], outcome_class=oc,
            definition=info.get("definition", ""), membership_test=info.get("membership_test", ""),
            exemplar_id=vs[0].id,
        )
    else:
        code = _dedup(f"{oc}_CLUSTER", modes)
        modes[code] = Mode(code=code, name=code, outcome_class=oc, exemplar_id=vs[0].id)
    return code


def _name_cluster(backend, vs: list[Verdict], oc: str, modes: dict[str, Mode], *, label: bool) -> str:
    """Label one cluster and register it (used by split/merge moves)."""
    info = backend.label([v.text for v in vs], oc) if label else None
    return _register_cluster(info, vs, oc, modes)


def relabel_all(clustering: Clustering, backend, *, workers: int = DEFAULT_WORKERS) -> Clustering:
    """Give every present mode a fresh semantic code + definition in one parallel
    batch. Used to label the *final* taxonomy after an LLM-free search, so the
    inner loop never spends model calls on candidates it will discard."""
    present = clustering.present_codes()
    members = {c: clustering.members(c) for c in present}
    infos = pmap(lambda c: backend.label([v.text for v in members[c]], clustering.outcome_of(c)),
                 present, workers)
    new_modes: dict[str, Mode] = {}
    remap: dict[str, str] = {}
    for c, info in zip(present, infos):
        remap[c] = _register_cluster(info, members[c], clustering.outcome_of(c), new_modes)
    new_assignment = {vid: remap[c] for vid, c in clustering.assignment.items() if c in remap}
    return Clustering(verdicts=clustering.verdicts, assignment=new_assignment, modes=new_modes)


def _dedup(code: str, modes: dict[str, Mode]) -> str:
    code = code.replace(" ", "_")
    if code not in modes:
        return code
    i = 2
    while f"{code}_{i}" in modes:
        i += 1
    return f"{code}_{i}"


class Geometry:
    """Embed all rationales once; reuse the vectors for every split/merge so the
    search loop never re-hits the embedding API."""

    def __init__(self, ids: list[str], X: np.ndarray):
        self.row = {vid: i for i, vid in enumerate(ids)}
        self.X = X

    @classmethod
    def build(cls, verdicts: list[Verdict], backend) -> "Geometry":
        ids = [v.id for v in verdicts]
        X = backend.embed([v.text for v in verdicts])
        return cls(ids, X)

    def subcluster(self, ids: list[str], k: int) -> list[list[str]]:
        rows = [self.row[i] for i in ids]
        sub = self.X[rows]
        labels = agglomerative(cosine_dist(sub), k)
        groups: dict[int, list[str]] = defaultdict(list)
        for vid, lab in zip(ids, labels):
            groups[int(lab)].append(vid)
        return [g for g in groups.values() if g]

    def centroid(self, ids: list[str]) -> np.ndarray:
        rows = [self.row[i] for i in ids]
        c = self.X[rows].mean(axis=0)
        n = np.linalg.norm(c) or 1.0
        return c / n

    def distance(self, ids_a: list[str], ids_b: list[str]) -> float:
        return float(1.0 - self.centroid(ids_a) @ self.centroid(ids_b))
