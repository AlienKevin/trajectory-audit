"""The load-bearing test: the optimizer must *raise* model-discriminativeness.

We plant a benchmark-driven coarse taxonomy (one TN bucket, NMI = 0) over data
whose two mechanisms are model-correlated. A correct optimizer splits the bucket
along the mechanism axis and recovers the model signal. This verifies the loop
independent of any real model's quality (mock backend, deterministic)."""
from failure_clustering.backends import MockBackend
from failure_clustering.cluster import Geometry
from failure_clustering.fitness import Evaluator, mi_stats
from failure_clustering.schema import Clustering, Mode, Verdict
from failure_clustering.search import hill_climb


MECH_A = "the solver was correct but its speedup stayed below the required threshold bar"
MECH_B = "the model never wrote the required output file so the deliverable was absent"


def planted_dataset():
    # mechanism A: 8 GPT, 2 GEM ; mechanism B: 2 GPT, 8 GEM  (model-correlated)
    verds = []
    for i in range(8):
        verds.append(Verdict(id=f"a_gpt_{i}", model="GPT", outcome_class="TN", text=MECH_A))
    for i in range(2):
        verds.append(Verdict(id=f"a_gem_{i}", model="GEM", outcome_class="TN", text=MECH_A))
    for i in range(2):
        verds.append(Verdict(id=f"b_gpt_{i}", model="GPT", outcome_class="TN", text=MECH_B))
    for i in range(8):
        verds.append(Verdict(id=f"b_gem_{i}", model="GEM", outcome_class="TN", text=MECH_B))
    return verds


def coarse_one_bucket(verds):
    mode = Mode(code="TN_ALL", name="All TN", outcome_class="TN")
    return Clustering(verdicts=verds, assignment={v.id: "TN_ALL" for v in verds},
                      modes={"TN_ALL": mode})


def test_hill_climb_raises_nmi_from_coarse_start():
    verds = planted_dataset()
    backend = MockBackend()
    geom = Geometry.build(verds, backend)
    ev = Evaluator(backend, geom=geom)

    initial = coarse_one_bucket(verds)
    nmi0 = mi_stats(initial).nmi_model
    assert nmi0 < 1e-9  # one bucket => mode tells you nothing about model

    final, history = hill_climb(initial, geom, ev, max_iters=8)
    nmi1 = mi_stats(final).nmi_model

    assert len(final.present_codes()) >= 2          # it split
    assert nmi1 > nmi0 + 0.1                         # and recovered model signal
    assert nmi1 > 0.15
    assert ev.scalar(final) > ev.scalar(initial)    # scalar improved
    # the split should align modes with mechanisms (each mode ~pure in mechanism)
    for c in final.present_codes():
        texts = {v.text for v in final.members(c)}
        assert len(texts) == 1


def test_hill_climb_is_monotone_and_terminates():
    verds = planted_dataset()
    backend = MockBackend()
    geom = Geometry.build(verds, backend)
    ev = Evaluator(backend, geom=geom)
    final, history = hill_climb(coarse_one_bucket(verds), geom, ev, max_iters=20)
    scores = [h["score"] for h in history]
    assert scores == sorted(scores)                 # never accepts a worsening move
    assert len(history) <= 21
