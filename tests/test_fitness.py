"""Fitness math: deterministic, no backend, no network."""
from failure_clustering.fitness import (
    Evaluator, adjusted_rand_index, leverage_norm, mdl_norm, mi_stats,
)
from failure_clustering.backends import MockBackend
from failure_clustering.schema import Clustering, Mode, Verdict


def build(rows):
    """rows: list of (id, model, outcome, mode_code)."""
    verdicts, assignment, modes = [], {}, {}
    for vid, model, oc, code in rows:
        verdicts.append(Verdict(id=vid, model=model, outcome_class=oc, text=f"{code} text {vid}"))
        assignment[vid] = code
        modes.setdefault(code, Mode(code=code, name=code, outcome_class=oc))
    return Clustering(verdicts=verdicts, assignment=assignment, modes=modes)


def test_perfect_discrimination_nmi_one():
    # mode == model: knowing the mode tells you the model exactly -> NMI 1.0
    rows = [(f"a{i}", "X", "TN", "M1") for i in range(4)] + \
           [(f"b{i}", "Y", "TN", "M2") for i in range(4)]
    mi = mi_stats(build(rows))
    assert mi.nmi_model > 0.99
    assert mi.mi > 0.99  # H(model)=1 bit, all of it recovered


def test_no_discrimination_nmi_zero():
    # every mode split evenly across models -> mode says nothing about model
    rows = []
    for c in ("M1", "M2"):
        rows += [(f"{c}x{i}", "X", "TN", c) for i in range(3)]
        rows += [(f"{c}y{i}", "Y", "TN", c) for i in range(3)]
    mi = mi_stats(build(rows))
    assert mi.nmi_model < 1e-9


def test_mdl_penalizes_more_modes():
    coarse = [(f"v{i}", "X" if i % 2 else "Y", "TN", "ONE") for i in range(12)]
    fine = [(f"v{i}", "X" if i % 2 else "Y", "TN", f"M{i}") for i in range(12)]
    assert mdl_norm(build(fine)) > mdl_norm(build(coarse))


def test_leverage_outcome_weighting():
    fp = build([(f"v{i}", "X", "FP", "C") for i in range(5)])
    tp = build([(f"v{i}", "X", "TP", "C") for i in range(5)])
    assert leverage_norm(fp) == 1.0      # FP weight is the max
    assert leverage_norm(tp) == 0.0      # TP weight is zero


def test_adjusted_rand_index():
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0  # label-invariant
    assert abs(adjusted_rand_index([0, 1, 0, 1], [0, 0, 1, 1])) < 0.6


def test_evaluator_scalar_is_finite():
    rows = [(f"a{i}", "X", "TN", "M1") for i in range(4)] + \
           [(f"b{i}", "Y", "TN", "M2") for i in range(4)]
    ev = Evaluator(MockBackend())
    fit = ev.fitness(build(rows))
    assert fit.scalar == fit.scalar  # not NaN
    assert fit.n_modes == 2
