"""Adapter: Harbor-Index audit output → generic ``Verdict`` list.

The aggregated pack (``verdicts.json``) already matches ``verdict.schema.json``,
so this is thin. It also normalises the long provider model ids
(``anthropic/claude-opus-4-8`` …) to short labels, which is what you want as the
discriminative target. Nothing here is imported by the core package — it is the
one Harbor-specific file, kept at the edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from failure_clustering.schema import Verdict, load_verdicts  # noqa: E402

MODEL_LABELS = {
    "anthropic/claude-opus-4-8": "Opus 4.8",
    "gpt-5.5": "GPT-5.5",
    "openai/gpt-5.5": "GPT-5.5",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1",
}


def short_model(agent_model: str) -> str:
    if agent_model in MODEL_LABELS:
        return MODEL_LABELS[agent_model]
    m = (agent_model or "").lower()
    if "opus" in m:
        return "Opus 4.8"
    if "gpt" in m:
        return "GPT-5.5"
    if "gem" in m:
        return "Gemini 3.1"
    return agent_model or "?"


def load(path: str | Path, *, text_field: str = "outcome_rationale") -> list[Verdict]:
    raw = load_verdicts(path, text_field=text_field)
    return [Verdict(id=v.id, model=short_model(v.model), outcome_class=v.outcome_class,
                    text=v.text, task=v.task, raw=v.raw) for v in raw]


if __name__ == "__main__":
    import sys
    vs = load(sys.argv[1])
    from collections import Counter
    print(f"{len(vs)} verdicts; models={dict(Counter(v.model for v in vs))}; "
          f"outcomes={dict(Counter(v.outcome_class for v in vs))}")
