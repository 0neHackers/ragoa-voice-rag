"""Chunking strategy interface and shared text utilities.

Every strategy is an independently testable object with the same signature, so the
index builder and the benchmark can swap them without knowing which one is in play —
that swappability is the point of the task's "don't submit one naive splitter" bar.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable

from data.loader import Example, Passage
from harness.types import Chunk

# Devanagari uses '।' (danda) as its full stop; a splitter that only knows '.' will
# treat an entire Hindi paragraph as one sentence and silently defeat both the
# semantic and recursive strategies.
SENTENCE_END = re.compile(r"(?<=[।?!.])\s+|\n{2,}")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
WHITESPACE = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    parts = [WHITESPACE.sub(" ", s).strip() for s in SENTENCE_END.split(text)]
    return [s for s in parts if s]


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in PARAGRAPH_BREAK.split(text)]
    return [p for p in parts if p]


def approx_tokens(text: str) -> int:
    """Cheap token estimate.

    A real tokenizer call per chunk-boundary decision would dominate index build time
    for a number that only needs to be roughly right. Devanagari runs ~2.5 chars/token
    under multilingual sentencepiece vocabularies versus ~4 for Latin script, so the
    estimate is script-aware rather than a flat chars/4.
    """
    if not text:
        return 0
    deva = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    ratio = 2.5 if deva > len(text) * 0.3 else 4.0
    return max(1, int(len(text) / ratio))


class ChunkingStrategy(ABC):
    """Base class. Subclasses implement `chunk_passage` or override `chunk_examples`."""

    name: str = "base"

    @abstractmethod
    def chunk_passage(self, passage: Passage) -> list[str]:
        """Split one passage's text into chunk strings."""

    def chunk_examples(self, examples: Iterable[Example]) -> list[Chunk]:
        """Apply the strategy across a corpus, attaching provenance metadata."""
        chunks: list[Chunk] = []
        for ex in examples:
            for passage in ex.passages:
                for i, text in enumerate(self.chunk_passage(passage)):
                    text = text.strip()
                    if not text:
                        continue
                    chunks.append(Chunk(
                        chunk_id=f"{self.name}:{ex.query_id}:{passage.passage_idx}:{i}",
                        text=text,
                        strategy=self.name,
                        query_id=ex.query_id,
                        passage_idx=passage.passage_idx,
                        source_lang=passage.source_lang,
                        char_len=len(text),
                        is_selected_passage=passage.is_selected,
                    ))
        return chunks

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "class": type(self).__name__}

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<{type(self).__name__} name={self.name!r}>"


__all__ = [
    "ChunkingStrategy", "split_sentences", "split_paragraphs",
    "approx_tokens", "SENTENCE_END",
]
