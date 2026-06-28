"""Command-line entry point.

    python -m failure_clustering optimize --verdicts PACK.json --backend openai --out result.json
    python -m failure_clustering score    --verdicts PACK.json [--result result.json]

``--backend mock`` runs fully offline (deterministic). ``--backend openai`` uses
any OpenAI-compatible endpoint via env (OPENROUTER_API_KEY, or OPENAI_API_KEY +
OPENAI_BASE_URL).
"""
from __future__ import annotations

import argparse
import json
import sys

from .backends import MockBackend, OpenAICompatBackend
from .fitness import Evaluator, mi_stats
from .optimize import optimize
from .schema import Clustering, Mode, load_verdicts


def _make_backend(args):
    if args.backend == "mock":
        return MockBackend()
    return OpenAICompatBackend.from_env(
        chat_model=args.model, embed_model=args.embed_model, batch=args.batch, workers=args.workers)


def cmd_optimize(args):
    verdicts = load_verdicts(args.verdicts, text_field=args.text_field)
    if not verdicts:
        sys.exit(f"no verdicts loaded from {args.verdicts}")
    backend = _make_backend(args)
    res = optimize(
        verdicts, backend,
        lenses=(args.lenses.split(",") if args.lenses else None),
        use_mechanism=not args.no_mechanism,
        use_coherence=not args.no_coherence,
        fidelity=not args.no_fidelity,
        max_iters=args.max_iters,
        critic_rounds=args.critic_rounds,
        workers=args.workers,
    )
    out = {
        "report": res.report,
        "taxonomy": res.clustering.to_dict(),
        "triage": res.triage,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    r = res.report
    print(f"\nNMI_model: {r['initial']['nmi_model']} -> {r['final']['nmi_model']} "
          f"(Δ {r['delta_nmi_model']:+})   modes: {r['initial']['n_modes']} -> {r['final']['n_modes']}")
    print(f"scalar:    {r['initial']['scalar']} -> {r['final']['scalar']} (Δ {r['delta_scalar']:+})")


def cmd_score(args):
    verdicts = load_verdicts(args.verdicts, text_field=args.text_field)
    if args.result:
        data = json.load(open(args.result))
        tax = data["taxonomy"] if "taxonomy" in data else data
        assignment = tax["assignment"]
        modes = {m["code"]: Mode(code=m["code"], name=m.get("name", m["code"]),
                                 outcome_class=m["outcome_class"], definition=m.get("definition", ""))
                 for m in tax["modes"]}
        clustering = Clustering(verdicts=verdicts, assignment=assignment, modes=modes)
    else:
        sys.exit("score needs --result with a saved taxonomy")
    mi = mi_stats(clustering)
    ev = Evaluator(MockBackend(), use_coherence=False)
    fit = ev.fitness(clustering)
    print(json.dumps({
        "n_rollouts": len(verdicts), "n_modes": fit.n_modes,
        "mi_bits": round(mi.mi, 4), "nmi_model": round(mi.nmi_model, 4),
        "cramers_v": round(mi.cramers_v, 4), "g": round(mi.g, 2), "dof": mi.dof,
    }, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="failure_clustering")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verdicts", required=True, help="verdict pack/glob/dir")
    common.add_argument("--text-field", default="outcome_rationale")

    o = sub.add_parser("optimize", parents=[common])
    o.add_argument("--backend", choices=["mock", "openai"], default="openai")
    o.add_argument("--model", default="openai/gpt-4o-mini")
    o.add_argument("--embed-model", default="openai/text-embedding-3-small")
    o.add_argument("--batch", type=int, default=24)
    o.add_argument("--workers", type=int, default=16)
    o.add_argument("--lenses", default=",".join(["failure mechanism", "root cause", "fix type"]))
    o.add_argument("--no-mechanism", action="store_true")
    o.add_argument("--no-coherence", action="store_true")
    o.add_argument("--no-fidelity", action="store_true")
    o.add_argument("--max-iters", type=int, default=30)
    o.add_argument("--critic-rounds", type=int, default=2)
    o.add_argument("--out")
    o.set_defaults(func=cmd_optimize)

    s = sub.add_parser("score", parents=[common])
    s.add_argument("--result", required=True)
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
