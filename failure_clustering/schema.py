"""Core data model for failure-mode clustering.

Everything downstream operates on a flat list of :class:`Verdict` records and a
mutable :class:`Clustering` (the optimizer's state). Nothing here is specific to
Harbor-Index — a Verdict is just *(id, model, outcome_class, free text)*. Use an
adapter (see ``adapters/``) to map your own audit format onto :class:`Verdict`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

# The only fixed partition. We cluster *within* each outcome class so a
# verifier-fault (FP/FN) can never share a bucket with an agent-fault (TN).
OUTCOME_CLASSES = ("TP", "TN", "FP", "FN")


@dataclass(frozen=True)
class Verdict:
    """One audited rollout, reduced to what clustering needs.

    ``text`` is the free-text the clusterer reads (a mechanism summary or the
    raw rationale). ``model`` is the discriminative target — the thing we want
    the taxonomy to be informative about. ``group`` is the hard stratification
    key (outcome class by default).
    """

    id: str
    model: str
    outcome_class: str
    text: str
    task: str | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def group(self) -> str:
        return self.outcome_class

    @classmethod
    def from_verdict_json(cls, d: dict, *, text_field: str = "outcome_rationale") -> "Verdict":
        """Build from a dict conforming to ``verdict.schema.json``."""
        jv = d.get("judge_verdict", {}) or {}
        text = d.get(text_field) or jv.get("summary") or ""
        return cls(
            id=d["rollout_id"],
            model=d.get("agent_model") or "?",
            outcome_class=d["outcome_class"],
            text=text,
            task=d.get("task_id"),
            raw=d,
        )


@dataclass
class Mode:
    """A named cluster: a falsifiable failure (or success) category."""

    code: str
    name: str
    outcome_class: str
    definition: str = ""
    membership_test: str = ""  # "belongs iff ..." — the falsifiable predicate
    exemplar_id: str | None = None
    color: str | None = None

    def copy(self) -> "Mode":
        return replace(self)


@dataclass
class Clustering:
    """Mutable optimizer state: assignment of verdicts to named modes.

    Invariant: every mode's ``outcome_class`` equals its members' outcome class
    (stratification is never crossed by any move).
    """

    verdicts: list[Verdict]
    assignment: dict[str, str]          # verdict.id -> mode.code
    modes: dict[str, Mode]              # mode.code -> Mode

    def __post_init__(self) -> None:
        self._by_id = {v.id: v for v in self.verdicts}

    # -- accessors -------------------------------------------------------
    def verdict(self, vid: str) -> Verdict:
        return self._by_id[vid]

    def members(self, code: str) -> list[Verdict]:
        return [self._by_id[vid] for vid, c in self.assignment.items() if c == code]

    def present_codes(self) -> list[str]:
        seen = {c for c in self.assignment.values()}
        return [c for c in self.modes if c in seen]

    def models(self) -> list[str]:
        return sorted({v.model for v in self.verdicts})

    def outcome_of(self, code: str) -> str:
        return self.modes[code].outcome_class

    def count_table(self) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
        """Return (codes, models, N[code][model]) over present modes."""
        codes = self.present_codes()
        models = self.models()
        N: dict[str, dict[str, int]] = {c: {m: 0 for m in models} for c in codes}
        for vid, code in self.assignment.items():
            if code in N:
                N[code][self._by_id[vid].model] += 1
        return codes, models, N

    def clone(self) -> "Clustering":
        return Clustering(
            verdicts=self.verdicts,
            assignment=dict(self.assignment),
            modes={k: v.copy() for k, v in self.modes.items()},
        )

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        codes, models, N = self.count_table()
        return {
            "n_rollouts": len(self.verdicts),
            "models": models,
            "modes": [
                {
                    "code": m.code,
                    "name": m.name,
                    "outcome_class": m.outcome_class,
                    "definition": m.definition,
                    "membership_test": m.membership_test,
                    "exemplar_id": m.exemplar_id,
                    "n": sum(N.get(m.code, {}).values()),
                    "counts": N.get(m.code, {}),
                }
                for m in self.modes.values()
                if m.code in codes
            ],
            "assignment": self.assignment,
        }


def load_verdicts(source: str | Path | Iterable[dict], *, text_field: str = "outcome_rationale") -> list[Verdict]:
    """Load verdicts from a JSON file/glob/dir, or an iterable of dicts.

    Accepts: a pack ``{"verdicts": [...]}``, a bare ``[...]`` list, a directory
    or glob of ``*.json`` verdict files, or any iterable of verdict dicts.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        records: list[dict] = []
        if p.is_dir():
            paths = sorted(p.glob("**/*.json"))
        elif any(ch in str(p) for ch in "*?[") or not p.exists():
            paths = sorted(Path().glob(str(p)))
        else:
            paths = [p]
        for fp in paths:
            obj = json.loads(fp.read_text())
            records.extend(_extract_records(obj))
        if not paths and p.exists():
            records.extend(_extract_records(json.loads(p.read_text())))
    else:
        records = list(source)
    return [Verdict.from_verdict_json(r, text_field=text_field) for r in records if r.get("outcome_class")]


def _extract_records(obj) -> list[dict]:
    if isinstance(obj, dict):
        if "verdicts" in obj:
            return list(obj["verdicts"])
        if "rollout_id" in obj and "outcome_class" in obj:
            return [obj]
        if "verdict" in obj and isinstance(obj["verdict"], dict):
            return [obj["verdict"]]
        return []
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    return []
