"""The optimization loop: greedy split/merge hill-climbing.

State = a :class:`Clustering`. Each move is accepted iff it raises the scalar
fitness. Splits chase model-discriminativeness (carve a benchmark-driven bucket
along a mechanism axis, which also tightens silhouette); merges remove clusters
that aren't separable. The inner loop is LLM-free — NMI comes from the count
table and silhouette from cached embeddings — so only the *final* taxonomy is
labelled. An adversarial critic proposes extra moves between hill-climbs and we
loop until it (and the hill-climb) run dry.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cluster import Geometry, _name_cluster, relabel_all
from .fitness import Evaluator
from .parallel import DEFAULT_WORKERS, pmap
from .schema import Clustering


@dataclass
class Move:
    op: str            # "split" | "merge"
    codes: list[str]
    result: Clustering
    score: float


def split_mode(clustering: Clustering, code: str, geom: Geometry, backend, k: int = 2,
               *, label: bool = False) -> Clustering | None:
    members = clustering.members(code)
    if len(members) < 4:
        return None
    groups = geom.subcluster([v.id for v in members], k)
    if len(groups) < 2:
        return None
    oc = clustering.outcome_of(code)
    new = clustering.clone()
    del new.modes[code]
    for g in groups:
        vs = [new.verdict(i) for i in g]
        newcode = _name_cluster(backend, vs, oc, new.modes, label=label)
        for v in vs:
            new.assignment[v.id] = newcode
    return new


def merge_modes(clustering: Clustering, a: str, b: str, backend, *, label: bool = False) -> Clustering | None:
    if clustering.outcome_of(a) != clustering.outcome_of(b):
        return None
    oc = clustering.outcome_of(a)
    new = clustering.clone()
    vs = new.members(a) + new.members(b)
    del new.modes[a]
    del new.modes[b]
    newcode = _name_cluster(backend, vs, oc, new.modes, label=label)
    for v in vs:
        new.assignment[v.id] = newcode
    return new


def _split_candidates(clustering: Clustering, ev: Evaluator, limit: int) -> list[str]:
    """Most internally dispersed modes first — most room to gain from a split.

    With geometry this is the total intra-cluster distance (free, deterministic);
    otherwise it falls back to size × (1 − LLM coherence)."""
    codes = [c for c in clustering.present_codes() if len(clustering.members(c)) >= 4]
    if ev.geom is not None:
        g = ev.geom

        def dispersion(c: str) -> float:
            ids = [v.id for v in clustering.members(c)]
            cen = g.centroid(ids)
            return float(sum(1.0 - g.X[g.row[i]] @ cen for i in ids))

        scored = [(dispersion(c), c) for c in codes]
    else:
        cohs = pmap(lambda c: ev._coherence_of(clustering.members(c)) if ev.use_coherence else 0.5,
                    codes, getattr(ev.backend, "workers", DEFAULT_WORKERS))
        scored = [(len(clustering.members(c)) * (1.0 - (coh or 0.5)), c) for c, coh in zip(codes, cohs)]
    scored.sort(reverse=True)
    return [c for _, c in scored[:limit]]


def _merge_candidates(clustering: Clustering, geom: Geometry, limit: int) -> list[tuple[str, str]]:
    """Closest mode pairs within each outcome stratum (by centroid distance)."""
    from collections import defaultdict
    by_oc: dict[str, list[str]] = defaultdict(list)
    for c in clustering.present_codes():
        by_oc[clustering.outcome_of(c)].append(c)
    pairs = []
    for oc, codes in by_oc.items():
        ids = {c: [v.id for v in clustering.members(c)] for c in codes}
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                d = geom.distance(ids[codes[i]], ids[codes[j]])
                pairs.append((d, codes[i], codes[j]))
    pairs.sort()
    return [(a, b) for _, a, b in pairs[:limit]]


def _allowed_splits(clustering: Clustering, cands: list[str], caps, max_modes) -> list[str]:
    """Drop split candidates that would breach a per-stratum cap or the global cap."""
    from collections import Counter
    if max_modes is not None and len(clustering.present_codes()) >= max_modes:
        return []
    if not caps:
        return cands
    strat = Counter(clustering.outcome_of(c) for c in clustering.present_codes())
    return [c for c in cands if strat[clustering.outcome_of(c)] < caps.get(clustering.outcome_of(c), 10 ** 9)]


def hill_climb(
    clustering: Clustering,
    geom: Geometry,
    ev: Evaluator,
    *,
    max_iters: int = 12,
    split_fanout: int = 4,
    merge_fanout: int = 4,
    eps: float = 1e-4,
    workers: int = DEFAULT_WORKERS,
    caps: dict | None = None,
    max_modes: int | None = None,
) -> tuple[Clustering, list[dict]]:
    cur = clustering
    cur_score = ev.scalar(cur)
    history = [{"iter": 0, "op": "init", "score": cur_score,
                "n_modes": len(cur.present_codes()),
                "nmi": ev.fitness(cur).nmi_model}]
    for it in range(1, max_iters + 1):
        # Build the candidate moves, then build+score them all concurrently —
        # each candidate is an independent clone, so this is embarrassingly parallel.
        split_cands = _allowed_splits(cur, _split_candidates(cur, ev, split_fanout), caps, max_modes)
        specs = [("split", c) for c in split_cands]
        specs += [("merge", pair) for pair in _merge_candidates(cur, geom, merge_fanout)]

        def evaluate(spec):
            op, arg = spec
            if op == "split":
                cand, codes = split_mode(cur, arg, geom, ev.backend, k=2), [arg]
            else:
                cand, codes = merge_modes(cur, arg[0], arg[1], ev.backend), list(arg)
            if cand is None:
                return None
            return Move(op, codes, cand, ev.scalar(cand))

        moves = [m for m in pmap(evaluate, specs, workers) if m is not None]
        best: Move | None = None
        for m in moves:  # input order preserved → deterministic tie-breaking
            if m.score > cur_score + eps and (best is None or m.score > best.score):
                best = m
        if best is None:
            break
        cur, cur_score = best.result, best.score
        history.append({"iter": it, "op": best.op, "codes": best.codes,
                        "score": cur_score, "n_modes": len(cur.present_codes()),
                        "nmi": ev.fitness(cur).nmi_model})
    return cur, history


def critic_loop(
    clustering: Clustering,
    geom: Geometry,
    ev: Evaluator,
    *,
    rounds: int = 2,
    workers: int = DEFAULT_WORKERS,
    caps: dict | None = None,
    max_modes: int | None = None,
    **hc_kwargs,
) -> tuple[Clustering, list[dict]]:
    """Hill-climb (LLM-free), label the result, ask the adversarial critic for
    fresh moves, repeat until the critic returns nothing new or no proposed move
    improves fitness."""
    cur = clustering
    history: list[dict] = []
    for r in range(rounds):
        cur, hc_hist = hill_climb(cur, geom, ev, workers=workers, caps=caps, max_modes=max_modes, **hc_kwargs)
        for h in hc_hist[1:]:
            h["round"] = r
        history.extend(hc_hist[1:] if r else hc_hist)
        # name the modes so the critic can reason about them
        cur = relabel_all(cur, ev.backend, workers=workers)
        moves = ev.backend.critique(_summarize(cur, ev))
        applied = False
        cur_score = ev.scalar(cur)
        for mv in moves:
            # respect the budget: a critic split into a capped/full stratum is dropped
            if mv.get("op") == "split" and mv.get("codes") and not _allowed_splits(
                    cur, [mv["codes"][0]], caps, max_modes):
                continue
            cand = _apply_critic_move(cur, mv, geom, ev.backend)
            if cand is None:
                continue
            s = ev.scalar(cand)
            if s > cur_score + 1e-4:
                cur, cur_score, applied = cand, s, True
                history.append({"round": r, "op": f"critic:{mv.get('op')}",
                                "codes": mv.get("codes"), "score": s,
                                "n_modes": len(cur.present_codes()),
                                "nmi": ev.fitness(cur).nmi_model})
        if not applied:
            break
    return cur, history


def _apply_critic_move(clustering, mv, geom, backend) -> Clustering | None:
    op = mv.get("op")
    codes = mv.get("codes") or []
    present = set(clustering.present_codes())
    if op == "split" and codes and codes[0] in present:
        return split_mode(clustering, codes[0], geom, backend, k=2)
    if op == "merge" and len(codes) >= 2 and codes[0] in present and codes[1] in present:
        return merge_modes(clustering, codes[0], codes[1], backend)
    return None


def _summarize(clustering: Clustering, ev: Evaluator) -> str:
    import json
    codes, models, N = clustering.count_table()
    rows = [{"code": c, "outcome": clustering.outcome_of(c),
             "n": sum(N[c].values()), "by_model": N[c],
             "definition": clustering.modes[c].definition} for c in codes]
    return json.dumps({"modes": rows}, ensure_ascii=False)
