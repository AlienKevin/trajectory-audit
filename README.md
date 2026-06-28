# Trajectory Audit

Trajectory Audit is a lightweight harness for auditing whether a benchmark verifier's PASS/FAIL signal was correct.

It turns one completed agent rollout into a new Harbor task. The judge model runs inside a fresh copy of the original task sandbox, reads the rollout artifacts under `/judge/`, and writes a grounded verdict to `/judge/verdict.json`.

## Core Idea

The verifier gives a binary signal:

| Verifier signal | Agent truly solved | Agent did not solve |
|---|---|---|
| PASS | `TP` | `FP` |
| FAIL | `FN` | `TN` |

The judge does not start from a taxonomy. It reads the task, trajectory, verifier output, reference solution, and tests, then decides whether the agent genuinely accomplished the task. Every claim must cite a trajectory step or exact file lines.

## Repository Layout

- `task_template/`: Harbor task template for one audit task.
- `task_template/instruction.md`: judge prompt.
- `task_template/tests/verify_verdict.py`: validates schema and citation grounding.
- `verdict.schema.json`: output contract for `TP/TN/FP/FN` verdicts.
- `judge_prompt_sections.py`: inserts PASS-specific or FAIL-specific decision guidance.
- `make_judge_task.py`: legacy generator for building audit tasks from stored Harbor rollouts.
- `aggregate_verdicts.py`: summarizes collected verdicts.
- `failure_clustering/`: generic bottom-up clustering + taxonomy optimization over collected verdicts (see below).
- `adapters/`: map a specific audit format onto the generic `Verdict` (e.g. `harbor_index.py`).
- `examples/`, `tests/`: worked example and the deterministic test suite.

## How To Run

1. Generate one audit task per rollout.

   ```bash
   python make_judge_task.py --trials <trial-id-or-task-name> --out tasks/audit
   ```

   The generator emits Harbor tasks shaped like:

   ```text
   tasks/audit/<rollout_id>/
     task.toml
     instruction.md
     environment/Dockerfile
     environment/judge/
     tests/
     solution/
   ```

2. Run the audit task with Harbor and your judge model.

   ```bash
   harbor run \
     -p tasks/audit \
     -a cursor-cli \
     -m cursor/composer-2.5 \
     -e daytona \
     -o runs/audit \
     --artifact /judge/verdict.json \
     -n 8 -y
   ```

3. Aggregate verdicts.

   ```bash
   python aggregate_verdicts.py 'runs/audit/**/artifacts/verdict.json' --out summary.json
   ```

## Verdict Contract

The judge writes `/judge/verdict.json` with:

- `outcome_class`: one of `TP`, `TN`, `FP`, `FN`
- `judge_verdict.task_truly_solved`: boolean
- `outcome_rationale`: concise explanation
- `evidence`: cited claims grounded in files or trajectory steps
- `verifier_or_task_concern`: optional note for verifier/task issues

The task verifier rejects verdicts that do not match `verdict.schema.json` or cite missing artifacts.

## Failure-Mode Clustering & Taxonomy Optimization

The judge deliberately emits **no taxonomy** — just `outcome_class` plus grounded
free text — so that failure modes can be discovered *bottom-up* from the verdicts.
The `failure_clustering/` package turns a pile of verdicts into a named taxonomy and
then **optimizes** that taxonomy to be maximally informative with minimal human
inspection. It is generic: it operates on `(id, model, outcome_class, rationale)`
records and knows nothing about Harbor.

### What it optimizes for

Clustering is unsupervised, so a scalar fitness drives the search:

| Term | Meaning | Why |
|---|---|---|
| **NMI(mode; model)** | fraction of "which model produced this" recoverable from the mode | the payoff — a taxonomy that doesn't separate models just re-describes the benchmark |
| **quality** | mean silhouette of the clusters in embedding space | has an *interior optimum*: collapsing a stratum into one bucket **and** over-splitting a tight blob both score badly. This is the guard a one-sided MDL term can't be — and it's deterministic, so the search needs no model calls |
| **modes** | mild penalty on mode count | parsimony tie-breaker |
| **leverage** | count × outcome weight (FP/FN weighted high) | up-weights verifier-fixable clusters |

`fitness = w_nmi·NMI + w_quality·silhouette − w_modes·(modes/N) + w_lev·leverage`

(MI, Cramér's V, and an MDL description-length are also reported for diagnostics.
A one-sided "fewer modes is always cheaper" MDL term was tried first and is exactly
why `quality` exists: it collapsed the whole true-negative stratum into a single
bucket. Silhouette has the two-sided optimum that prevents that.)

### The loop

1. **Mechanism, not surface** — each rationale is compressed to a benchmark-agnostic
   mechanism clause, so clusters form around *how it failed*, not *which dataset*.
2. **Stratify by `outcome_class`** (hard constraint) — a spurious-pass FP can never
   land in an agent-fault TN bucket.
3. **Consensus clustering** — agglomerate under several embedding "lenses"
   (mechanism / root-cause / fix-type) and fuse via a co-assignment matrix; the
   single biggest lever against embedding/LLM nondeterminism.
4. **Greedy split/merge hill-climb** — split a benchmark-driven bucket along a
   mechanism axis (raises NMI and tightens silhouette), merge two clusters that
   aren't separable. Each move accepted iff it raises fitness. The inner loop is
   LLM-free (NMI from the count table, silhouette from cached embeddings); only the
   *final* taxonomy is sent to the model for naming.
5. **Adversarial critic, loop-until-dry** — the model proposes split/merge/reassign
   moves; apply the improving ones; repeat until nothing new survives.
6. **Round-trip fidelity + boundary triage** — score each mode by re-predicting its
   members from the definition alone; surface only the lowest-silhouette boundary
   rollouts for a human (≈10, not all N).

### Backends

Everything depends only on the `LLMBackend` interface:

- `MockBackend` — deterministic, offline (numpy tf-idf + token heuristics). The test
  suite verifies the optimization *math and control flow* with it (planted
  model-discriminative structure is recovered; moves are monotone in fitness).
- `OpenAICompatBackend` — any OpenAI-compatible endpoint (OpenRouter, OpenAI, a local
  vLLM). Real embeddings + chat. Configured from env (`OPENROUTER_API_KEY`, or
  `OPENAI_API_KEY` + `OPENAI_BASE_URL`).

### Run it

```bash
pip install -r requirements.txt
python -m pytest -q                       # deterministic verification, no network

# Optimize a taxonomy (any OpenAI-compatible key):
export OPENROUTER_API_KEY=sk-or-...
python -m failure_clustering optimize \
  --verdicts 'runs/audit/**/artifacts/verdict.json' \
  --backend openai --model openai/gpt-4o-mini \
  --max-modes 16 --focus TN --out result.json

# Offline dry run / CI:
python -m failure_clustering optimize --verdicts verdicts.json --backend mock --out result.json
```

**Budget the taxonomy.** `--max-modes` is a hard cap on total modes; `--focus`
gives one outcome class (e.g. `TN`, genuine agent failures) most of that budget,
while the rest collapse to a few each — pass `outcome_caps={"TP": 1}` (the
`examples/` runner does this) to make true-positives a single "solved" bucket.

**Mechanism model matters most.** The single most important quality lever is the
model that writes the one-clause mechanism summaries (`--mechanism-model`, separate
from the cheap labeling model). A weak model emits generic clauses ("failed to meet
the requirement") that collapse every true-negative into one bucket; a strong one
(`openai/gpt-4o`, `google/gemini-2.5-pro`) emits distinct mechanisms that cluster
cleanly. On the Harbor-Index audit, swapping `gpt-4o-mini → gpt-4o` for this step
alone raised cluster silhouette 0.37 → 0.78 and the bottom-up TN modes converged on
the hand-written taxonomy (threshold-missed, missing-output, wrong-algorithm,
incomplete-fix, wrong-method, wrong-choice, refusal).

`result.json` holds the report (initial → final NMI / modes / fitness, the move
history), the optimized `taxonomy` (modes with definitions, falsifiable membership
tests, exemplars, per-model counts, and the assignment), and the human triage queue.

### Use your own data

Implement a one-function adapter to `failure_clustering.schema.Verdict` (see
`adapters/harbor_index.py`), or feed any iterable of dicts shaped like
`verdict.schema.json` to `failure_clustering.load_verdicts`.

## Notes

- This repo contains harness code only. It intentionally excludes historical run outputs, trajectories, task payloads, and generated audit artifacts.
- `make_judge_task.py` reflects the original Harbor-Index workspace layout and may need path adaptation for a different rollout archive.
- The audit model should be treated as another agent run: it can inspect and execute in the sandbox, but every conclusion must be grounded in cited artifacts.
