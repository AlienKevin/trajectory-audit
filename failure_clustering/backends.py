"""LLM backends behind one interface.

The whole pipeline depends only on :class:`LLMBackend`. Two implementations ship:

* :class:`MockBackend` — deterministic, offline (numpy only). Used by the test
  suite so the *control flow and the optimization math* are verifiable without a
  network or any model nondeterminism.
* :class:`OpenAICompatBackend` — any OpenAI-compatible endpoint (OpenRouter,
  OpenAI, a local vLLM, …). This is the real backend.

Add your own by implementing the same methods. Nothing else in the package
knows which backend it is talking to.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from . import features
from .parallel import pmap


@runtime_checkable
class LLMBackend(Protocol):
    """Everything the clusterer/optimizer asks of a model."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) array of unit-norm row embeddings for clustering."""

    def mechanism(self, texts: Sequence[str]) -> list[str]:
        """Compress each rationale to one *benchmark-agnostic* mechanism clause."""

    def label(self, texts: Sequence[str], outcome_class: str) -> dict:
        """Name a cluster. Return {name, code, definition, membership_test}."""

    def coherence(self, texts: Sequence[str]) -> float:
        """0..1 — do these rationales share a single failure mechanism?"""

    def distinct(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> float:
        """0..1 — are these two clusters mechanistically distinct?"""

    def predict_membership(self, definition: str, texts: Sequence[str]) -> list[bool]:
        """Given only a mode's definition, predict membership (round-trip fidelity)."""

    def critique(self, summary: str) -> list[dict]:
        """Adversarial critic. Return move dicts: {op: split|merge|reassign, ...}."""


# ---------------------------------------------------------------------------
# Mock backend — deterministic, offline
# ---------------------------------------------------------------------------
class MockBackend:
    """Deterministic stand-in. Embeddings are tf-idf; semantic judgments are
    token-overlap heuristics. Lets tests assert the optimizer behaves correctly
    independent of any real model's quality."""

    name = "mock"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return features.tfidf_embed(list(texts))

    def mechanism(self, texts: Sequence[str]) -> list[str]:
        return [features.first_clause(t) for t in texts]

    def _top_tokens(self, texts: Sequence[str], k: int = 4) -> list[str]:
        from collections import Counter
        c: Counter = Counter()
        for t in texts:
            c.update(set(features.tokenize(t)))
        return [tok for tok, _ in c.most_common(k)]

    def label(self, texts: Sequence[str], outcome_class: str) -> dict:
        toks = self._top_tokens(texts)
        code = ("_".join(toks[:3]) or "cluster").upper()
        name = " ".join(t.capitalize() for t in toks[:3]) or "Cluster"
        return {
            "name": name,
            "code": code,
            "definition": f"Rollouts characterised by: {', '.join(toks) or 'n/a'}.",
            "membership_test": f"belongs iff the rationale centres on {', '.join(toks[:2]) or 'n/a'}",
        }

    def coherence(self, texts: Sequence[str]) -> float:
        if len(texts) < 2:
            return 1.0
        X = features.tfidf_embed(list(texts))
        S = X @ X.T
        n = len(texts)
        off = (S.sum() - np.trace(S)) / (n * (n - 1))
        return float(np.clip(off, 0.0, 1.0))

    def distinct(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> float:
        X = features.tfidf_embed(list(texts_a) + list(texts_b))
        a = X[: len(texts_a)].mean(axis=0)
        b = X[len(texts_a):].mean(axis=0)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.clip(1.0 - (a @ b) / denom, 0.0, 1.0))

    def predict_membership(self, definition: str, texts: Sequence[str]) -> list[bool]:
        key = set(features.tokenize(definition))
        out = []
        for t in texts:
            toks = set(features.tokenize(t))
            j = len(key & toks) / (len(key | toks) or 1)
            out.append(j >= 0.08)
        return out

    def critique(self, summary: str) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# OpenAI-compatible backend — real models (OpenRouter, OpenAI, vLLM, …)
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class OpenAICompatBackend:
    """Talks to any OpenAI-compatible Chat Completions + Embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        chat_model: str = "openai/gpt-4o-mini",
        embed_model: str = "openai/text-embedding-3-small",
        mechanism_model: str | None = None,
        embed_dim_fallback: int = 1536,
        max_retries: int = 4,
        batch: int = 24,
        workers: int = 8,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat_model = chat_model
        self.mechanism_model = mechanism_model or chat_model
        self.embed_model = embed_model
        self.embed_dim_fallback = embed_dim_fallback
        self.max_retries = max_retries
        self.batch = batch
        self.workers = workers
        self.name = f"openai-compat:{chat_model}"

    @classmethod
    def from_env(cls, **kw) -> "OpenAICompatBackend":
        """Prefer OPENROUTER_API_KEY, else OPENAI_API_KEY (+ OPENAI_BASE_URL)."""
        if os.environ.get("OPENROUTER_API_KEY"):
            return cls(api_key=os.environ["OPENROUTER_API_KEY"],
                       base_url="https://openrouter.ai/api/v1", **kw)
        return cls(api_key=os.environ["OPENAI_API_KEY"],
                   base_url=os.environ.get("OPENAI_BASE_URL") or None, **kw)

    # -- low level -------------------------------------------------------
    def _chat(self, system: str, user: str, *, model: str | None = None) -> str:
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.chat.completions.create(
                    model=model or self.chat_model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0,
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - transient API errors
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"chat failed after {self.max_retries} retries: {last}")

    def _chat_json(self, system: str, user: str, *, default, model: str | None = None):
        raw = self._chat(system + " Respond with JSON only, no prose.", user, model=model)
        m = _JSON_RE.search(raw)
        if not m:
            return default
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return default

    # -- interface -------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            chunk = [t or " " for t in texts[i: i + 256]]
            r = self._client.embeddings.create(model=self.embed_model, input=chunk)
            vecs.extend(d.embedding for d in r.data)
        X = np.asarray(vecs, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def mechanism(self, texts: Sequence[str]) -> list[str]:
        texts = list(texts)
        sys = ("Distil one audit rationale into a single clause naming the SPECIFIC "
               "underlying failure mechanism — what *kind* of thing went wrong and why "
               "(for instance: a performance threshold missed, an incomplete or wrongly "
               "targeted fix, a wrong algorithm or formula, a misread figure/document, a "
               "missing or ungraded output, a wrong discrete choice, a refusal). Be precise "
               "about the TYPE of error, but abstract away the benchmark, file names and "
               "numbers so that similar mechanisms across different tasks share wording.")

        def do(chunk: list[str]) -> list[str]:
            payload = [{"i": j, "rationale": t} for j, t in enumerate(chunk)]
            res = self._chat_json(
                sys,
                "Return JSON {\"items\":[{\"i\":int,\"mechanism\":str}...]} for:\n"
                + json.dumps(payload, ensure_ascii=False),
                default={"items": []},
                model=self.mechanism_model,
            )
            got = {d.get("i"): d.get("mechanism", "") for d in res.get("items", []) if isinstance(d, dict)}
            return [got.get(j) or features.first_clause(chunk[j]) for j in range(len(chunk))]

        batches = [texts[i: i + self.batch] for i in range(0, len(texts), self.batch)]
        out: list[str] = []
        for r in pmap(do, batches, self.workers):
            out.extend(r)
        return out

    def label(self, texts: Sequence[str], outcome_class: str) -> dict:
        sample = list(texts)[:30]
        sys = ("You name a cluster of agent-rollout failure rationales. Give a "
               "SCREAMING_SNAKE code, a short Title Case name, a one-sentence "
               "definition of the shared mechanism, and a falsifiable membership "
               "test of the form 'belongs iff ...'.")
        res = self._chat_json(
            sys,
            f"outcome_class={outcome_class}. Rationales:\n" + json.dumps(sample, ensure_ascii=False)
            + '\nReturn {"code":...,"name":...,"definition":...,"membership_test":...}',
            default={},
        )
        toks = " ".join(sample[:1]).split()[:3]
        return {
            "code": str(res.get("code") or ("_".join(toks) or "CLUSTER")).upper().replace(" ", "_"),
            "name": str(res.get("name") or "Cluster"),
            "definition": str(res.get("definition") or ""),
            "membership_test": str(res.get("membership_test") or ""),
        }

    def coherence(self, texts: Sequence[str]) -> float:
        if len(texts) < 2:
            return 1.0
        res = self._chat_json(
            "Rate 0..1 how strongly these rationales share ONE failure mechanism.",
            json.dumps(list(texts)[:30], ensure_ascii=False) + '\nReturn {"score":float}',
            default={"score": 0.5},
        )
        return float(np.clip(res.get("score", 0.5), 0.0, 1.0))

    def distinct(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> float:
        res = self._chat_json(
            "Rate 0..1 how mechanistically DISTINCT cluster A is from cluster B "
            "(1 = different mechanisms, 0 = same).",
            "A:\n" + json.dumps(list(texts_a)[:15], ensure_ascii=False)
            + "\nB:\n" + json.dumps(list(texts_b)[:15], ensure_ascii=False)
            + '\nReturn {"score":float}',
            default={"score": 0.5},
        )
        return float(np.clip(res.get("score", 0.5), 0.0, 1.0))

    def predict_membership(self, definition: str, texts: Sequence[str]) -> list[bool]:
        texts = list(texts)

        def do(chunk: list[str]) -> list[bool]:
            res = self._chat_json(
                "Given a cluster DEFINITION, decide for each rationale whether it "
                "belongs. Judge only against the definition.",
                f"definition: {definition}\nrationales:\n"
                + json.dumps([{"i": j, "t": t} for j, t in enumerate(chunk)], ensure_ascii=False)
                + '\nReturn {"items":[{"i":int,"belongs":bool}...]}',
                default={"items": []},
            )
            got = {d.get("i"): bool(d.get("belongs")) for d in res.get("items", []) if isinstance(d, dict)}
            return [got.get(j, False) for j in range(len(chunk))]

        batches = [texts[i: i + self.batch] for i in range(0, len(texts), self.batch)]
        out: list[bool] = []
        for r in pmap(do, batches, self.workers):
            out.extend(r)
        return out

    def critique(self, summary: str) -> list[dict]:
        res = self._chat_json(
            "You are an adversarial critic of a failure-mode taxonomy. Propose "
            "moves that increase coherence/distinctness. Ops: split (a code whose "
            "members span >1 mechanism), merge (two codes a fix wouldn't tell "
            "apart), reassign. Only high-confidence moves.",
            summary + '\nReturn {"moves":[{"op":...,"codes":[...],"reason":...}...]}',
            default={"moves": []},
        )
        return [m for m in res.get("moves", []) if isinstance(m, dict) and m.get("op")]
