"""optimize() runs end-to-end on the mock backend and never worsens fitness."""
from failure_clustering.backends import MockBackend
from failure_clustering.optimize import optimize
from failure_clustering.schema import Verdict

MECHS = {
    "TN": [
        "the solver was correct but the measured speedup fell below the required threshold",
        "the code crashed on an empty input due to a logic defect in the pipeline",
        "the agent never wrote the required output file the deliverable was absent",
    ],
    "FN": [
        "the work was correct but the grading harness broke on a sandbox error",
        "the answer was right but an over-strict acceptance gate rejected it",
    ],
    "TP": ["the agent genuinely solved the task and the pass is real"],
    "FP": ["the agent oracle-hacked the answer so the pass is spurious"],
}


def synthetic():
    verds, n = [], 0
    # model-correlated: GEM gets more 'missing deliverable', GPT more 'threshold'
    plan = [
        ("TN", 0, "GPT", 6), ("TN", 0, "GEM", 2),
        ("TN", 1, "GPT", 3), ("TN", 1, "GEM", 3),
        ("TN", 2, "GPT", 1), ("TN", 2, "GEM", 6),
        ("FN", 0, "OPUS", 4), ("FN", 1, "OPUS", 3), ("FN", 1, "GPT", 1),
        ("TP", 0, "GPT", 5), ("TP", 0, "GEM", 3), ("TP", 0, "OPUS", 4),
        ("FP", 0, "GEM", 1),
    ]
    for oc, idx, model, count in plan:
        for _ in range(count):
            verds.append(Verdict(id=f"v{n}", model=model, outcome_class=oc, text=MECHS[oc][idx]))
            n += 1
    return verds


def test_optimize_runs_and_does_not_worsen():
    verds = synthetic()
    res = optimize(verds, MockBackend(), max_iters=8, critic_rounds=1)
    r = res.report

    # never worsens the scalar objective
    assert r["final"]["scalar"] >= r["initial"]["scalar"] - 1e-9

    # output is a valid, outcome-pure clustering covering every rollout
    cl = res.clustering
    assert set(cl.assignment) == {v.id for v in verds}
    for c in cl.present_codes():
        assert len({v.outcome_class for v in cl.members(c)}) == 1
        assert cl.outcome_of(c) == cl.members(c)[0].outcome_class

    # report + triage shape
    assert "fidelity" in r and "history" in r
    assert res.triage and "silhouette" in res.triage[0]
    assert r["final"]["n_modes"] >= 4  # at least one mode per populated outcome class


def test_max_modes_and_tp_singleton_respected():
    verds = synthetic()
    res = optimize(verds, MockBackend(), max_iters=10, critic_rounds=1, fidelity=False,
                   max_modes=8, focus_outcome="TN", outcome_caps={"TP": 1})
    cl = res.clustering
    assert len(cl.present_codes()) <= 8                       # global cap honoured
    tp_modes = [c for c in cl.present_codes() if cl.outcome_of(c) == "TP"]
    assert len(tp_modes) == 1                                 # TP collapsed to one bucket
    # focus stratum (TN) gets the largest share of the budget
    from collections import Counter
    by_oc = Counter(cl.outcome_of(c) for c in cl.present_codes())
    assert by_oc["TN"] == max(by_oc.values())


def test_serialization_roundtrips():
    res = optimize(synthetic(), MockBackend(), max_iters=4, critic_rounds=1, fidelity=False)
    d = res.clustering.to_dict()
    assert d["n_rollouts"] == len(res.clustering.verdicts)
    assert sum(m["n"] for m in d["modes"]) == d["n_rollouts"]
