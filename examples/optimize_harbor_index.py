#!/usr/bin/env python3
"""Worked example: optimize a failure-mode taxonomy for Harbor-Index verdicts.

    export OPENROUTER_API_KEY=sk-or-...        # or OPENAI_API_KEY + OPENAI_BASE_URL
    python examples/optimize_harbor_index.py path/to/verdicts.json --out result.json

This is the only Harbor-specific entry point; the package itself is generic.
Swap ``harbor_index.load`` for your own adapter to run on any audit dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters import harbor_index
from failure_clustering.backends import MockBackend, OpenAICompatBackend
from failure_clustering.optimize import optimize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts", help="Harbor verdicts.json pack")
    ap.add_argument("--backend", choices=["openai", "mock"], default="openai")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--critic-rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16, help="concurrent backend calls")
    ap.add_argument("--max-modes", type=int, default=16, help="hard cap on total modes")
    ap.add_argument("--focus", default="TN", help="outcome class that gets most of the budget")
    ap.add_argument("--mechanism-model", default="openai/gpt-4o",
                    help="stronger model just for mechanism extraction — the one "
                         "quality-critical step (a weak model collapses distinct "
                         "failure mechanisms into one bucket). ~9 calls; cheap.")
    args = ap.parse_args()

    verds = harbor_index.load(args.verdicts)
    print(f"loaded {len(verds)} verdicts", flush=True)
    backend = (MockBackend() if args.backend == "mock"
               else OpenAICompatBackend.from_env(chat_model=args.model, workers=args.workers,
                                                 mechanism_model=args.mechanism_model))

    res = optimize(verds, backend, max_iters=args.max_iters,
                   critic_rounds=args.critic_rounds, workers=args.workers,
                   max_modes=args.max_modes, focus_outcome=args.focus,
                   outcome_caps={"TP": 1})  # TP is a single "genuinely solved" bucket

    Path(args.out).write_text(json.dumps({
        "report": res.report,
        "taxonomy": res.clustering.to_dict(),
        "triage": res.triage,
    }, indent=2))

    r = res.report
    print(f"\n=== {r['backend']} | {r['n_rollouts']} rollouts ===")
    print(f"NMI_model: {r['initial']['nmi_model']} -> {r['final']['nmi_model']} "
          f"(Δ {r['delta_nmi_model']:+})")
    print(f"modes:     {r['initial']['n_modes']} -> {r['final']['n_modes']}")
    print(f"quality:   {r['initial']['quality']} -> {r['final']['quality']}")
    print(f"scalar:    {r['initial']['scalar']} -> {r['final']['scalar']} (Δ {r['delta_scalar']:+})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
