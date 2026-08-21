"""Grounded answer generation.

Two modes, and the response always says which one produced it:

- `generated` — Claude Haiku 4.5 synthesises an answer from the retrieved passages.
- `extractive` — no LLM key configured, so the highest-scoring retrieved span is returned
  verbatim.

The extractive mode exists so the retrieval pipeline, the guardrails, and the whole
benchmark stay runnable and honestly measurable without an LLM key (DECISIONS.md D6). It
is labelled in `Answer.mode` and surfaced by the demo — a copied passage is never
presented as a generated answer.

`NO_ANSWER` from the model is not an error. It is the model using the refusal the prompt
authorises, and it is converted into the same typed decline the guardrails produce, so
"the corpus doesn't cover this" reaches the user identically no matter which component
noticed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from generation.prompts import NO_ANSWER_SENTINEL, build_messages
from harness.types import Answer, ScoredChunk

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

#: Kept tight on purpose. The prompt asks for two or three sentences because the answer is
#: read aloud, and output tokens are the dominant term in generation latency.
MAX_TOKENS = 300


class NoAnswerFromModel(RuntimeError):
    """The model declined, per the prompt's `NO_ANSWER` instruction."""


@dataclass(slots=True)
class GeneratorConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.0   # grounded extraction, not creative writing
    timeout_s: float = 25.0

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            max_tokens=int(os.getenv("GENERATION_MAX_TOKENS", str(MAX_TOKENS))),
        )


class Generator:
    """Turns a question plus retrieved chunks into an `Answer`."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig.from_env()

    @property
    def mode(self) -> str:
        return "generated" if self.config.api_key else "extractive"

    def generate(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        """Raises on transport failure — the orchestrator's stage wrapper types it."""
        if not chunks:
            raise NoAnswerFromModel("No retrieved passages to ground an answer in.")

        if not self.config.api_key:
            return self._extractive(chunks)

        return self._generate_anthropic(question, chunks)

    # -- LLM path ---------------------------------------------------------- #

    def _generate_anthropic(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        import requests

        system_prompt, user_message = build_messages(question, chunks)

        response = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=self.config.timeout_s,
        )

        if response.status_code in (401, 403):
            raise PermissionError(f"Anthropic rejected the API key (HTTP {response.status_code}).")
        if response.status_code == 429:
            raise RuntimeError("429 rate limited by Anthropic.")
        if response.status_code >= 500:
            raise RuntimeError(f"Anthropic {response.status_code}: {response.text[:200]}")
        if response.status_code >= 400:
            raise ValueError(f"Anthropic {response.status_code}: {response.text[:300]}")

        body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        ).strip()

        if not text:
            raise RuntimeError("Anthropic returned an empty completion.")

        if text.strip().upper().startswith(NO_ANSWER_SENTINEL):
            raise NoAnswerFromModel(
                "The model reported that the retrieved passages do not answer this question."
            )

        return Answer(
            text=text,
            mode="generated",
            model=self.config.model,
            cited_chunk_ids=_resolve_citations(text, chunks),
        )

    # -- extractive fallback ------------------------------------------------ #

    def _extractive(self, chunks: list[ScoredChunk]) -> Answer:
        """Return the best passage's leading sentences, verbatim.

        Ranked by dense cosine rather than fused rank, for the same reason the confidence
        gate is: the fused top hit can be a lexical-only match that shares a rare token
        with the query without being about it.
        """
        best = max(
            chunks,
            key=lambda c: c.dense_score if c.dense_score is not None else c.score,
        )
        return Answer(
            text=_leading_sentences(best.chunk.text, limit=400),
            mode="extractive",
            model="none (extractive fallback — set ANTHROPIC_API_KEY to enable generation)",
            cited_chunk_ids=[best.chunk.chunk_id],
        )


_SENTENCE_END = re.compile(r"(?<=[।.!?])\s+")


def _leading_sentences(text: str, limit: int = 400) -> str:
    """First whole sentences up to `limit` chars. Splits on Devanagari danda too."""
    text = text.strip()
    if len(text) <= limit:
        return text

    out: list[str] = []
    used = 0
    for sentence in _SENTENCE_END.split(text):
        if used + len(sentence) > limit and out:
            break
        out.append(sentence)
        used += len(sentence) + 1
    return " ".join(out).strip() or text[:limit].rstrip()


def _resolve_citations(text: str, chunks: list[ScoredChunk]) -> list[str]:
    """Map the model's `[n]` markers back to chunk ids.

    Out-of-range markers are dropped rather than clamped — a citation to a passage that
    was not supplied is a hallucinated citation, and pointing it at a real chunk would
    manufacture evidence the model never saw.
    """
    ids: list[str] = []
    for marker in re.findall(r"\[(\d{1,2})\]", text):
        idx = int(marker) - 1
        if 0 <= idx < len(chunks):
            chunk_id = chunks[idx].chunk.chunk_id
            if chunk_id not in ids:
                ids.append(chunk_id)
    return ids


__all__ = ["Generator", "GeneratorConfig", "NoAnswerFromModel", "DEFAULT_MODEL"]
