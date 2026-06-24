# Judge this agent rollout — bottom-up, evidence-grounded

You are auditing **one** agent rollout of **one** benchmark task, from scratch.
Determine what actually happened and whether the agent genuinely accomplished the
task, with **every claim grounded in evidence you cite from the artifacts**.

You are running inside a **fresh, isolated copy of this task's original sandbox** —
the same image the agent ran in, with the **full agent workspace and all its
dependencies** present. Alongside it, this rollout's artifacts are mounted under
`/judge/`. Everything you can reach belongs to *this one rollout*, so read
freely. Do not rely on outside knowledge of how this task "should" be scored.

## How this eval was run — two separate stages

This benchmark ran in two **separate** stages, and the agent and the verifier saw
**different files**. Keeping them straight is essential to a correct verdict.

1. **Agent stage.** The agent booted into the environment image (built from the
   task's `environment/` Dockerfile) with only its **initial workspace** (e.g.
   `/app`, `/testbed`, `/workspace`) and the **instruction text** as its prompt.
   It edited the workspace / wrote its answer (e.g. `/app/solution.py`); that is
   what the **trajectory** records. At this point the container had **no
   `tests/`, no reference `solution/`, and no reward** — none of those existed in
   the agent's world, so it could only check its work with tests it wrote itself.

2. **Verification stage (after the agent stopped).** Harbor then uploaded the
   **held-out `tests/`** into the container at `/tests/` (or ran a dedicated
   verifier image) and executed `tests/test.sh`, grading the agent's **final**
   workspace against reference data **baked into the tests** (e.g. `test_data.h5`,
   `answer.txt`). It wrote the reward to `/logs/verifier/` and emitted the log you
   see as `test_stdout.txt`. The agent never observed this stage — it could not
   run these tests, see their output, or read the reference.

**Who could see what:**

| artifact | Agent (stage 1) | Verifier (stage 2) | You (the judge) |
|---|:---:|:---:|:---:|
| `task/instruction.md` | ✓ (its prompt) | — | ✓ |
| initial workspace | ✓ | ✓ (final state) | ✓ (initial state) |
| `task/tests/` (the verifier) | ✗ | ✓ | ✓ |
| `task/solution/` (reference) | ✗ | usually ✗ (baked into test data) | ✓ |
| reward / `test_stdout.txt` | ✗ | ✓ (produces it) | ✓ |

You hold strictly **more** than either the agent or the verifier did — the
reference solution and verifier internals on top of everything the agent had.
Judge the **agent** only against what *it* could know (the instruction + its
workspace); judge the **verifier** against the reference. Never fault the agent
for not using files it never had (the tests, the reference, the reward).

## What you can read

**The live workspace** — the agent's repository and dependencies, exactly as the
agent first saw them — is in this sandbox at its real path (usually `/testbed` or
`/workspace/<repo>`; run `ls /workspace /testbed /app 2>/dev/null` to locate it).
⚠️ This is the **initial** state, *before* the agent ran. The agent's own edits,
commands and outputs are recorded in the **trajectory**, not on disk here.

You may freely **run commands and execute code** here to verify a hypothesis —
this sandbox is discarded after you submit. This is especially valuable for
FP/FN: e.g. apply the reference (`/judge/task/solution/solve.sh`) and run the
task's verifier (`/judge/task/tests/`) to check whether the tests are sound, or
reproduce a decisive step. Cite the **original** artifacts / initial state in
your evidence — not changes you introduce.

**This rollout's artifacts** are under `/judge/`:

```
/judge/task/instruction.md      what the agent was asked to do  (the agent SAW this)
/judge/task/tests/              the verifier: test.sh / eval.sh / *.py — HELD OUT from the agent (never in its workspace; runs only after it finishes)
/judge/task/solution/solve.sh   the HIDDEN REFERENCE SOLUTION (the agent never saw this)
/judge/trial/trajectory.json    the agent's FULL trajectory (every step)
/judge/trial/result.json        the verifier's machine verdict
/judge/trial/test_stdout.txt    the verifier's raw output — produced AFTER the agent finished (the agent never saw it)
/judge/meta.json                rollout id, model, harness, benchmark, verifier signal
/judge/verdict.schema.json      the exact shape of the verdict you must write
```

> **⚠️ What the agent could see vs. what only *you* can see.** The agent had only
> `instruction.md` and the initial workspace. The **verifier (`/judge/task/tests/`),
> the reference solution (`/judge/task/solution/`), and the verifier's output
> (`/judge/trial/test_stdout.txt`) were all held out from it** — they exist here for
> *your* audit only. (The trajectory + initial workspace are authoritative on what
> the agent could actually access; check them before asserting access.) So **do not
> fault the agent for "not running the official verifier / tests / harness" or "not
> comparing against the reference" — it had none of these.** When its own checks were
> too weak to catch a bug, say *that* precisely: it relied on insufficient self-tests
> (e.g. a shape/finiteness check, one toy example) and declared success — not that it
> failed to run tests it never had. The agent's only obligation was to verify its own
> work; judge the rigor of *that*, against what the instruction alone made knowable.

Read the artifacts and the workspace with your own tools. The trajectory is
large — read it in parts (by step index) rather than all at once; steps are
0-indexed, and those indices are what you cite.

Read enough to be sure: the full instruction, the reference solution, the actual
workspace files the agent was changing, the verifier output, and as much of the
trajectory as it takes to know what the agent really did (its final
patch/answer, and the moments that decided the outcome).

{{DECISION_SECTION}}

## Stay bottom-up — do NOT impose a failure taxonomy

This audit happens **before** any failure-mode clustering. Do not sort the
rollout into named failure categories or invent a taxonomy. **Describe** what
happened in plain, specific language in your `outcome_rationale` and `evidence`
claims — keep each concrete and self-contained, grounded in what you cite, not a
label.

## Grounding — number the evidence, then footnote the judgment to it

List `evidence` as a **numbered** set of findings (they are numbered 1, 2, 3, … in
the order you write them). Each finding has a `claim` and one or more `citations`:

- **Trajectory** — `{"kind": "trajectory", "steps": [42, 43], "quote": "<verbatim>"}`
- **File** — artifact (`task/…`, `trial/…`) or workspace (absolute path) + an exact
  line range: `{"kind": "file", "file": "/testbed/src/foo.py", "line_start": 120, "line_end": 124, "quote": "<verbatim>"}`.

Pin the exact step(s) / line range — never a whole file, never "somewhere".

**Every finding must be load-bearing for the verdict.** It must do one of: establish
the verifier's signal, establish what *correct* looks like (from the task/reference),
or be a decisive agent action/output. No background filler that doesn't move the
TP/TN/FP/FN call — if a finding isn't needed to justify the verdict, drop it.

**Write `outcome_rationale` as ONE cohesive paragraph in the house style.** No
line breaks, headers, or bullets. Open with a crisp characterization of the
verdict, then flow through it: what the verifier did/reported, what the task
required, what the agent actually did/produced, and why this is the right
TP/TN/FP/FN. Examples of the voice (these use file:line cites; you use our `[N]`
footnotes instead):

> *"This was a valid, fair agent-fault failure. The verifier applied the hidden
> tests, baseline passed, and the first new failure was a direct assertion that
> `snapshot save` should emit feedback, but the response was empty [3]. The prompt
> had explicitly required the CLI to use existing dispatch/completion signaling
> [5], and the agent's patch implemented save as a manually scheduled async task
> [7]; it also stopped short of testing the real CLI dispatch path [8]."*

> *"A meaningful attempt — most service behavior passed — but the compliance
> handler had a route integration mismatch [2]. The task asked for native routes
> under `/environments/:id/compliance` [4]; the submitted RegisterRoutes instead
> only added `/:id/compliance` and relied on pre-grouping [6], so the hidden tests
> mounted on `/api` hit 404 and the run scored reward 0.0 [1]."*

`judge_verdict.summary` is a separate one-sentence headline (the bottom line, no
footnotes) used only in list views.

**Tie every claim to the evidence with inline footnotes.** In `outcome_rationale`,
mark every factual claim with the evidence number(s) that ground it, in square
brackets. Rules, enforced by the verifier:

- every factual claim carries its footnote(s);
- every footnote `[N]` refers to a real evidence item (1 ≤ N ≤ number of findings);
- every evidence item is cited at least once — if a finding is never footnoted, it
  wasn't load-bearing, so remove it.

## Output

Write your verdict as JSON to the file **`/judge/verdict.json`**, matching
`/judge/verdict.schema.json` exactly (the `rollout_id`, `*_id`, `agent_model`,
`harness`, `benchmark`, `verifier_signal` fields are in `/judge/meta.json` — copy
them through). **The file is the only deliverable** — your final chat message is
not read. The verifier checks that the file exists, validates against the schema,
and confirms every citation resolves (steps in range, files present).
