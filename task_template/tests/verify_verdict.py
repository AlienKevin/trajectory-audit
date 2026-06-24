#!/usr/bin/env python3
"""
Judge-task verifier. Reward = the judge produced a well-formed, grounded verdict.

Checks /judge/verdict.json:
  1. valid JSON and validates against /judge/verdict.schema.json;
  2. every citation RESOLVES — trajectory steps are in range, cited files exist
     (artifacts under /judge, or workspace files at absolute paths) and the line
     range is sane.
Publishes the verdict to /logs/verifier/verdict.json for collection, then exits
0 (pass) or 1 (fail). The CORRECTNESS of the judgment is evaluated later by the
human-agreement study — this gate only enforces well-formedness + grounding.
"""
import json, os, sys
from pathlib import Path

JUDGE = Path("/judge")
LOGS = Path("/logs/verifier"); LOGS.mkdir(parents=True, exist_ok=True)

def fail(msg):
    print(f"VERDICT INVALID: {msg}"); sys.exit(1)

def main():
    vp = JUDGE / "verdict.json"
    if not vp.is_file():
        fail("no /judge/verdict.json was produced")
    try:
        verdict = json.loads(vp.read_text())
    except Exception as e:
        fail(f"verdict.json is not valid JSON: {e}")

    schema = json.loads((JUDGE / "verdict.schema.json").read_text())
    try:
        import jsonschema
        jsonschema.validate(verdict, schema)
    except ImportError:
        for k in schema.get("required", []):
            if k not in verdict: fail(f"missing required field: {k}")
    except Exception as e:
        fail(f"schema validation failed: {e}")

    # citation resolvability
    steps = json.loads((JUDGE / "trial/trajectory.json").read_text())
    steps = steps.get("steps", steps) if isinstance(steps, dict) else steps
    nsteps = len(steps)
    for ev in verdict.get("evidence", []):
        for c in ev.get("citations", []):
            if c.get("kind") == "trajectory":
                for s in c.get("steps", []):
                    if not 0 <= s < nsteps:
                        fail(f"trajectory step {s} out of range 0..{nsteps-1}")
            elif c.get("kind") == "file":
                if c["line_start"] > c["line_end"]:
                    fail(f"line_start>line_end for {c['file']}")
                f = c["file"]
                p = JUDGE / f if not f.startswith("/") else Path(f)   # artifact or workspace
                if not p.is_file():
                    fail(f"cited file does not exist: {f}")

    # footnote linkage: every [N] in the summary/rationale points at a real
    # evidence item, and every evidence item is footnoted at least once (so each
    # is load-bearing and the judgment is tied to its grounding).
    import re
    nev = len(verdict.get("evidence", []))
    prose = verdict["judge_verdict"]["summary"] + " " + verdict["outcome_rationale"]
    refs = {int(m) for m in re.findall(r"\[(\d+)\]", prose)}
    bad = sorted(n for n in refs if not 1 <= n <= nev)
    if bad:
        fail(f"footnote(s) {bad} reference no evidence item (have 1..{nev})")
    uncited = [i for i in range(1, nev + 1) if i not in refs]
    if uncited:
        fail(f"evidence item(s) {uncited} are never footnoted — every finding must be load-bearing")

    (LOGS / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"VERDICT OK: {verdict['outcome_class']} "
          f"(truly_solved={verdict['judge_verdict']['task_truly_solved']}, "
          f"{len(verdict.get('evidence', []))} grounded claims)")
    sys.exit(0)

if __name__ == "__main__":
    main()
