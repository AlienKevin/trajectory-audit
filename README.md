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

## Notes

- This repo contains harness code only. It intentionally excludes historical run outputs, trajectories, task payloads, and generated audit artifacts.
- `make_judge_task.py` reflects the original Harbor-Index workspace layout and may need path adaptation for a different rollout archive.
- The audit model should be treated as another agent run: it can inspect and execute in the sandbox, but every conclusion must be grounded in cited artifacts.
