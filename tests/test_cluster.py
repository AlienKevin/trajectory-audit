"""Clustering primitives: agglomerative, stratification, geometry."""
import numpy as np

from failure_clustering.backends import MockBackend
from failure_clustering.cluster import Geometry, agglomerative, build_clustering
from failure_clustering.features import cosine_dist
from failure_clustering.schema import Verdict


def test_agglomerative_recovers_two_blobs():
    # two well-separated blobs in 2-D
    X = np.array([[1, 0]] * 5 + [[0, 1]] * 5, dtype=float)
    X += np.array([[0.01 * (i % 3), 0.0] for i in range(10)])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = agglomerative(cosine_dist(X), 2)
    assert len(set(labels)) == 2
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1


def test_agglomerative_k_bounds():
    X = np.eye(6)
    assert len(set(agglomerative(cosine_dist(X), 1))) == 1
    assert len(set(agglomerative(cosine_dist(X), 6))) == 6
    assert len(set(agglomerative(cosine_dist(X), 99))) == 6  # capped at n


def test_build_clustering_respects_outcome_strata():
    verds = [Verdict(id=f"tn{i}", model="X", outcome_class="TN", text=f"speed threshold below bar {i}") for i in range(6)]
    verds += [Verdict(id=f"fn{i}", model="Y", outcome_class="FN", text=f"verifier harness broke {i}") for i in range(6)]
    cl = build_clustering(verds, MockBackend())
    # every mode is pure in outcome class
    for code in cl.present_codes():
        ocs = {v.outcome_class for v in cl.members(code)}
        assert len(ocs) == 1 and cl.outcome_of(code) in ("TN", "FN")
    # TN and FN never share a mode
    assert all(cl.outcome_of(c) == "TN" for c in cl.present_codes() if c.startswith("TN"))


def test_geometry_subcluster_splits_distinct_text():
    verds = [Verdict(id=f"a{i}", model="X", outcome_class="TN", text="speedup threshold benchmark below required bar") for i in range(4)]
    verds += [Verdict(id=f"b{i}", model="Y", outcome_class="TN", text="never wrote the required output file deliverable absent") for i in range(4)]
    geom = Geometry.build(verds, MockBackend())
    groups = geom.subcluster([v.id for v in verds], 2)
    assert len(groups) == 2
    # the two text families separate
    a_ids = {f"a{i}" for i in range(4)}
    for g in groups:
        s = set(g)
        assert s <= a_ids or s.isdisjoint(a_ids)
