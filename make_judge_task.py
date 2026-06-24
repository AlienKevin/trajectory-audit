#!/usr/bin/env python3
"""
Generate Harbor judge-task(s) — one per rollout.

Each emitted task is a normal Harbor task the harness runs like any benchmark:
its environment is a THIN image `FROM` the rollout's original task image (so the
full agent workspace + deps are present) with the rollout's artifacts COPY'd to
/judge/. The harness's own agent (the judge model, set by run-config — the task
is model-agnostic) reads /judge/ + the workspace and writes /judge/verdict.json;
tests/ validates it. Run the SAME tasks with 3 judge models for the agreement
study.

    python make_judge_task.py --select one-per-task --n 20 --seed 42 --out tasks/sel20
    python make_judge_task.py --trials trial-11 --out tasks/smoke

Then point the harness at tasks/<...>/<rollout_id>/ with --model opus|gpt-5.5|composer.
"""
from __future__ import annotations
import argparse, json, random, shutil, tomllib
from pathlib import Path

from judge_prompt_sections import render_judge_instruction

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRIALS_ROOT = REPO / "harbor-index-trials" / "trials_extracted"
DATASET_ROOTS = [REPO / "task_dataset" / "harbor-index" / "datasets" / "daytona",
                 REPO / "task_dataset" / "harbor-index" / "datasets" / "modal",
                 HERE / "tasks" / "tb3_agent20"]  # staged TB3-preview task defs
DATASET_FLAT = REPO / "task_dataset" / "hae-index-src" / "harbor-index" / "datasets"
META_JSONL = REPO / "aft_compare" / "scripts" / "_runner_meta.jsonl"
TEMPLATE = HERE / "task_template"

TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "bottom-up rollout judge"
category = "meta-judge"
tags = ["judge", "failure-analysis", "{benchmark}"]
# rollout under judgement (provenance)
judged_rollout = "{rollout_id}"

[verifier]
timeout_sec = 600

[agent]
# time budget for the JUDGE model (the judge harness/model is a run-config, not pinned here)
timeout_sec = {agent_timeout}

[environment]
# Built from environment/Dockerfile (thin layer FROM the original task image).
# If your harness requires a prebuilt image, build+push this and set docker_image.
docker_image = ""
build_timeout_sec = 1800
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
allow_internet = true          # the in-sandbox judge must reach the model API
"""

DOCKERFILE = """\
# Thin judge layer over the rollout's ORIGINAL environment, so the live agent
# workspace + all dependencies are present exactly as the agent saw them.
FROM {base_image}

{python_bootstrap}

# This rollout's artifacts (read-only inputs for the judge).
COPY judge/ /judge/
"""

PYTHON_BOOTSTRAP = """\
# The judge verifier is Python-based, but some original task images are bare
# Ubuntu/alpine images. Keep the layer no-op when python3 is already present.
RUN if ! command -v python3 >/dev/null 2>&1; then \\
      if command -v apt-get >/dev/null 2>&1; then \\
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*; \\
      elif command -v apk >/dev/null 2>&1; then \\
        apk add --no-cache python3 py3-pip ca-certificates; \\
      elif command -v microdnf >/dev/null 2>&1; then \\
        microdnf install -y python3 python3-pip ca-certificates && microdnf clean all; \\
      else \\
        echo 'No supported package manager found to install python3' >&2; exit 1; \\
      fi; \\
    fi
"""


def _norm(r):
    """_runner_meta.jsonl has two export shapes; normalize to one."""
    return {
        "trial_id": r.get("trial_id"),
        "task_name": r.get("task_name") or r.get("task"),
        "requested_task_name": r.get("requested_task_name"),
        "task_path": r.get("task_path", ""),
        "model": r.get("model"),
        "agent": r.get("agent") or r.get("harness"),
    }

def load_meta():
    return [_norm(json.loads(l)) for l in META_JSONL.read_text().splitlines() if l.strip()]

_UPLOADED = None
def real_verifier_signal(trial_id, fallback):
    """Real reward/threshold from harbor-mix's uploaded_trials.jsonl. Continuous-
    reward tasks (AlgoTune etc.) gate on a CUSTOM threshold, so a 1.03x speedup is a
    genuine FAIL even though pytest passes — surface the real score vs threshold
    instead of a defaulted/null reward. Falls back when the trial isn't in the
    upload (e.g. the stratified-failure sample, which is reward=0 by construction)."""
    global _UPLOADED
    if _UPLOADED is None:
        _UPLOADED = {}
        up = REPO / "harbor-index-trials" / "uploaded_trials.jsonl"
        if up.is_file():
            for l in up.read_text().splitlines():
                if l.strip():
                    u = json.loads(l); _UPLOADED[u["trial_id"]] = u
    u = _UPLOADED.get(trial_id)
    if not u:
        return fallback
    sm, sc, th = u.get("score_mode"), u.get("raw_reward_eff", u.get("score")), u.get("threshold")
    if u.get("force_fail"):
        binpass = 0
    elif sm == "threshold" and th is not None:
        binpass = 1 if (sc is not None and sc >= th) else 0
    else:
        binpass = 1 if (sc is not None and sc >= 1.0) else 0
    metric = sm
    if sm == "threshold" and th is not None:
        metric = f"threshold mode — continuous score {sc:g} vs pass threshold {th:g}"
    return {"binary_reward": binpass, "reward": sc,
            "status": "PASS" if binpass else "FAIL", "reward_metric": metric}

def resolve_task_dir(row):
    """Mirror harbor_judge.resolve_task_source: trials use 'aa-lcr_aa-lcr-18',
    the dataset dir is 'aa-lcr-aa-lcr-18', and task_path is a third name — so try
    several candidates, then fall back to matching task.toml [task] name."""
    cands = []
    if n := row.get("requested_task_name"):
        cands.append(n)
        if "/" in n:
            cands.append(n.replace("/", "-"))
    if n := row.get("task_name"):
        cands.append(n)
    if tp := row.get("task_path", ""):
        cands.append(tp.rstrip("/").rsplit("/", 1)[-1])
    for n in cands:
        for base in DATASET_ROOTS + [DATASET_FLAT]:
            if (base / n).is_dir():
                return base / n
    # fallback: [task] name inside task.toml (handles gaia2 on-disk renames)
    import re
    want = row.get("requested_task_name") or row.get("task_name")
    for base in DATASET_ROOTS + [DATASET_FLAT]:
        if not base.is_dir():
            continue
        for d in base.iterdir():
            tt = d / "task.toml"
            if tt.is_file() and re.search(rf'(?m)^\s*name\s*=\s*"{re.escape(want or "")}"', tt.read_text()):
                return d
    return None

def select(args):
    rows = load_meta()
    if args.trials:
        want = set(args.trials.split(","))
        rows = [r for r in rows if r["trial_id"] in want or r["task_name"] in want]
    if args.select == "one-per-task":
        rnd = random.Random(args.seed)
        by_task = {}
        for r in rows:
            by_task.setdefault(r["task_name"], []).append(r)
        rows = [rnd.choice(v) for v in by_task.values()]
        rnd.shuffle(rows); rows = rows[: args.n]
    return rows


def emit(row, out_root, agent_timeout) -> str | None:
    task_dir = resolve_task_dir(row)
    trial_dir = TRIALS_ROOT / row["task_name"] / row["trial_id"]
    rollout_id = f"{row['task_name']}__{row['trial_id']}"
    if not task_dir or not trial_dir.is_dir():
        print(f"  skip {rollout_id}: unresolved source"); return None

    env = tomllib.loads((task_dir / "task.toml").read_text()).get("environment", {})
    orig_env = task_dir / "environment"
    result = json.loads((trial_dir / "result.json").read_text())
    meta = {
        "rollout_id": rollout_id, "task_id": row["task_name"], "trial_id": row["trial_id"],
        "agent_model": row.get("model"), "harness": row.get("agent"),
        "benchmark": row["task_name"].split("_")[0].split("-")[0],
        "verifier_signal": real_verifier_signal(row["trial_id"], {
            "binary_reward": int(result.get("binary_reward") or 0), "reward": result.get("reward"),
            "status": result.get("status"), "reward_metric": result.get("reward_metric")}),
    }

    dst = out_root / rollout_id
    (dst / "environment").mkdir(parents=True, exist_ok=True)
    (dst / "tests").mkdir(exist_ok=True)
    (dst / "solution").mkdir(exist_ok=True)

    # task.toml + Dockerfile
    (dst / "task.toml").write_text(TASK_TOML.format(
        benchmark=meta["benchmark"], rollout_id=rollout_id, agent_timeout=agent_timeout,
        cpus=env.get("cpus", 2), memory_mb=env.get("memory_mb", 4096), storage_mb=env.get("storage_mb", 8192)))
    # Reuse the task's ORIGINAL environment build (its public-base Dockerfile +
    # COPY context) so Daytona rebuilds the real workspace WITHOUT needing creds
    # for the private prebuilt ghcr image; then append our read-only judge layer.
    if orig_env.is_dir():
        shutil.copytree(orig_env, dst / "environment", dirs_exist_ok=True)
    df = dst / "environment" / "Dockerfile"
    if df.is_file():
        df.write_text(df.read_text().rstrip()
                      + "\n\n# --- bottom-up judge verifier runtime ---\n"
                      + PYTHON_BOOTSTRAP
                      + "\n# --- bottom-up judge artifacts (read-only inputs) ---\nCOPY judge/ /judge/\n")
    else:
        df.write_text(DOCKERFILE.format(
            base_image=env.get('docker_image') or 'python:3.12-slim',
            python_bootstrap=PYTHON_BOOTSTRAP.rstrip()))

    # static template files
    (dst / "instruction.md").write_text(
        render_judge_instruction(TEMPLATE / "instruction.md", meta["verifier_signal"]["binary_reward"])
    )
    shutil.copy(TEMPLATE / "tests" / "test.sh", dst / "tests" / "test.sh")
    shutil.copy(TEMPLATE / "tests" / "verify_verdict.py", dst / "tests" / "verify_verdict.py")
    shutil.copy(TEMPLATE / "solution" / "solve.sh", dst / "solution" / "solve.sh")

    # build context: judge/ -> /judge/ in the image
    j = dst / "environment" / "judge"
    (j / "trial").mkdir(parents=True, exist_ok=True)
    for f in ("trajectory.json", "result.json", "test_stdout.txt"):
        if (trial_dir / f).exists():
            shutil.copy(trial_dir / f, j / "trial" / f)
    (j / "task").mkdir(exist_ok=True)
    if (task_dir / "instruction.md").exists():
        shutil.copy(task_dir / "instruction.md", j / "task" / "instruction.md")
    if (task_dir / "task.toml").exists():
        shutil.copy(task_dir / "task.toml", j / "task" / "task.toml")
    for sub in ("tests", "solution"):                       # original verifier + reference, for the judge to read
        if (task_dir / sub).is_dir():
            shutil.copytree(task_dir / sub, j / "task" / sub, dirs_exist_ok=True)
    (j / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(HERE / "verdict.schema.json", j / "verdict.schema.json")
    return rollout_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["all", "one-per-task"], default="all")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", default="")
    ap.add_argument("--agent-timeout", type=int, default=3600)
    ap.add_argument("--out", default="tasks/judge")
    args = ap.parse_args()

    rows = select(args)
    out_root = (HERE / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    emitted = [r for r in (emit(row, out_root, args.agent_timeout) for row in rows) if r]
    (out_root / "manifest.json").write_text(json.dumps(emitted, indent=2))
    print(f"emitted {len(emitted)} judge tasks -> {out_root}")
    print("run each with your harness, model = a run-config (opus | gpt-5.5 | composer).")


if __name__ == "__main__":
    raise SystemExit(main())
