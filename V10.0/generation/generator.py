"""Grounded answer generation, on Sarvam.

Two modes, and the response always says which one produced it:

- `generated` — `sarvam-105b-conversations` writes an answer from the retrieved passages.
- `extractive` — no key configured, so the highest-scoring retrieved span comes back
  verbatim, labelled as such.

**Why Sarvam and not a second provider.** This build originally called Claude Haiku here.
That meant two vendors, two keys, two failure modes, and two billing relationships for a
pipeline whose speech-to-text, translation and text-to-speech all already run on Sarvam.
Consolidating removes a whole class of "which key is missing" problems and keeps the demo
alive on one credential.

**Model choice was measured, not assumed.** Sarvam exposes two chat models and they behave
very differently here:

    sarvam-105b                 25.5s   865 completion tokens   emits reasoning_content
    sarvam-105b-conversations    2.3s    21 completion tokens   answers directly

`sarvam-105b` is a reasoning model. On a grounded-extraction task — where the answer is
sitting in the context and the job is to state it — that reasoning is pure latency. Worse,
it spends the token budget thinking: at `max_tokens=160` it returned
`finish_reason: "length"` with `content: null` and the entire budget in `reasoning_content`.
`reasoning_effort: "low"` did not fix it. So the conversations variant is the default, and
`_extract_content` still checks `reasoning_content` so a misconfiguration surfaces as a
clear error instead of a mystery empty answer.

`NO_ANSWER` from the model is not an error. It's the model using the refusal the prompt
authorises, and it becomes the same typed decline the guardrails produce, so "the corpus
doesn't cover this" reaches the user identically no matter which component noticed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from generation.prompts import NO_ANSWER_SENTINEL, build_messages
from harness.types import Answer, ScoredChunk

#: The non-reasoning variant. See the module docstring for the measurement behind this.
DEFAULT_MODEL = "sarvam-105b-conversations"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

#: The prompt asks for two or three sentences because the answer gets read aloud, and
#: output tokens dominate generation latency.
MAX_TOKENS = 300


class NoAnswerFromModel(RuntimeError):
    """The model declined, per the prompt's `NO_ANSWER` instruction."""


@dataclass(slots=True)
class GeneratorConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.0   # grounded extraction, not creative writing
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        return cls(
            api_key=os.getenv("SARVAM_API_KEY", "").strip(),
            model=os.getenv("SARVAM_CHAT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
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

        return self._generate_sarvam(question, chunks)

    # -- LLM path ---------------------------------------------------------- #

    def _generate_sarvam(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        import requests

        system_prompt, user_message = build_messages(question, chunks)

        response = requests.post(
            SARVAM_CHAT_URL,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=self.config.timeout_s,
        )

        if response.status_code in (401, 403):
            raise PermissionError(f"Sarvam rejected the API key (HTTP {response.status_code}).")
        if response.status_code == 429:
            raise RuntimeError("429 rate limited by Sarvam.")
        if response.status_code >= 500:
            raise RuntimeError(f"Sarvam {response.status_code}: {response.text[:200]}")

        # Quota exhaustion is an account state, not a malformed request, and letting it
        # error out would take a working retrieval pipeline down with it. Degrade to
        # extractive and say why, in the answer itself.
        if response.status_code == 400 and _is_quota_error(response.text):
            answer = self._extractive(chunks)
            answer.model = (
                "extractive fallback — Sarvam quota exhausted, so no text was generated; "
                "this is a retrieved passage verbatim"
            )
            return answer

        if response.status_code >= 400:
            raise ValueError(f"Sarvam {response.status_code}: {response.text[:300]}")

        text = _extract_content(response.json())

        if _is_refusal(text):
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
            model="none (extractive fallback — set SARVAM_API_KEY to enable generation)",
            cited_chunk_ids=[best.chunk.chunk_id],
        )


def _extract_content(body: dict) -> str:
    """Pull the answer out of a chat completion, and explain an empty one.

    A reasoning model that runs out of budget returns `content: null` with everything in
    `reasoning_content` and `finish_reason: "length"`. That's a configuration problem with
    a specific fix, so it gets a specific message rather than a bare "empty completion".
    """
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Sarvam returned no choices.")

    choice = choices[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    if text:
        return text

    if message.get("reasoning_content"):
        raise RuntimeError(
            f"Model {body.get('model')!r} spent its whole token budget on reasoning and "
            f"returned no answer (finish_reason={choice.get('finish_reason')!r}). Use "
            f"'{DEFAULT_MODEL}' — the non-reasoning variant — or raise "
            "GENERATION_MAX_TOKENS well above the reasoning length."
        )

    raise RuntimeError(
        f"Sarvam returned an empty completion (finish_reason={choice.get('finish_reason')!r})."
    )


def _is_refusal(text: str) -> bool:
    """Detect the authorised refusal wherever the model puts it.

    The prompt asks for `NO_ANSWER` alone, but models routinely explain themselves first
    and append it: "The passages do not state how many... NO_ANSWER". Matching only at
    the start meant those refusals fell through as if they were answers, and then got
    rejected downstream for the wrong reason.

    Matched as a standalone token so an answer that merely discusses the sentinel is not
    swallowed by it.
    """
    return re.search(rf"{NO_ANSWER_SENTINEL}", (text or "").upper()) is not None


def _is_quota_error(body: str) -> bool:
    """Distinguish 'out of quota' from 'your request was malformed'.

    Both arrive as HTTP 400. Only the first is worth degrading gracefully for — a genuinely
    malformed request is our bug and should surface loudly.
    """
    low = (body or "").lower()
    return any(s in low for s in ("quota", "credit", "billing", "insufficient", "exceeded"))


_SENTENCE_END = re.compile(r"(?<=[।.!?])\s+")


def _leading_sentences(text: str, limit: int = 400) -> str:
    """First whole sentences up to `limit` chars. Splits on the Devanagari danda too."""
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
    wasn't supplied is a hallucinated citation, and pointing it at a real chunk would
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
