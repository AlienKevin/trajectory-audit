#!/bin/bash
# Reference "solution" for a judge task: emit a minimal, schema-valid, grounded
# verdict. There is no gold judgment, so this only demonstrates the task is
# solvable (well-formed output is achievable) — it is not a correct judgment.
set -euo pipefail

python3 - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("/judge/meta.json").read_text())
verdict = {
    **{k: meta.get(k) for k in ("rollout_id","task_id","trial_id","agent_model","harness","benchmark")},
    "verifier_signal": meta["verifier_signal"],
    "judge_verdict": {
        "task_truly_solved": bool(meta["verifier_signal"]["binary_reward"]),
        "confidence": "low",
        "summary": "Reference stub verdict mirroring the verifier signal [1] (well-formedness demo, not a real judgment).",
    },
    "outcome_class": "TP" if meta["verifier_signal"]["binary_reward"] else "TN",
    "outcome_rationale": "Stub: mirrors the verifier's recorded reward [1]; not an independent judgment.",
    "evidence": [{
        "claim": "The verifier recorded this binary_reward.",
        "citations": [{"kind": "file", "file": "trial/result.json", "line_start": 1, "line_end": 1,
                       "quote": "binary_reward"}],
    }],
    "verifier_or_task_concern": None,
}
Path("/judge/verdict.json").write_text(json.dumps(verdict, indent=2))
print("wrote stub /judge/verdict.json")
PY
