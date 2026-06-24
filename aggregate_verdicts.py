#!/usr/bin/env python3
"""
Roll up collected judge verdicts into a summary.

Point it at wherever your harness drops the per-run verdicts (each a verdict.json
emitted to /logs/verifier/ by the verifier). Works across providers and runs.

    python aggregate_verdicts.py 'runs/**/verdict.json' --out summary.json

Outputs overall + per-(model, benchmark, judge-provider) TP/TN/FP/FN counts and
the verifier-disagreement rate (FP+FN)/n — the headline "how often the reward is
misleading" number, and the input to the human↔judge agreement study.
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="glob(s) for verdict.json files")
    ap.add_argument("--out", default="summary.json")
    args = ap.parse_args()

    paths = [p for pat in args.patterns for p in glob.glob(pat, recursive=True)]
    verdicts = []
    for p in paths:
        try:
            verdicts.append(json.loads(Path(p).read_text()))
        except Exception as e:
            print(f"  skip {p}: {e}")

    cells = lambda: {c: 0 for c in ("TP", "TN", "FP", "FN")}
    overall = cells()
    by = {k: defaultdict(cells) for k in ("agent_model", "benchmark", "_judge_provider")}
    for v in verdicts:
        c = v.get("outcome_class")
        if c not in overall:
            continue
        overall[c] += 1
        for k in by:
            by[k][v.get(k, "?")][c] += 1

    n = sum(overall.values())
    summary = {
        "n_verdicts": n,
        "outcome_counts": overall,
        "verifier_disagreement_rate": round((overall["FP"] + overall["FN"]) / n, 3) if n else None,
        "by_agent_model": {k: dict(v) for k, v in by["agent_model"].items()},
        "by_benchmark":   {k: dict(v) for k, v in by["benchmark"].items()},
        "by_judge":       {k: dict(v) for k, v in by["_judge_provider"].items()},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({"n": n, **overall, "disagreement(FP+FN)": summary["verifier_disagreement_rate"]}, indent=1))

if __name__ == "__main__":
    raise SystemExit(main())
