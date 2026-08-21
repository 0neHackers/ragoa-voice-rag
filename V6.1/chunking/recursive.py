"""Strategy 3 — recursive, structure-aware chunking.

Splits on the most meaningful boundary that works, and only falls back to a cruder one
when a piece is still too large: paragraph → sentence → clause → hard character cut.

The difference from fixed-size is that a cut lands *between* units of meaning rather
than through one. The difference from semantic chunking is cost: this is pure string
work with no embedding calls, so it chunks a corpus in milliseconds rather than
seconds — which makes it the sensible default when index build time matters.

Small adjacent pieces are packed back up to the target size afterwards. Without that
step, structure-aware splitting on short paragraphs produces a swarm of tiny chunks
that individually carry too little context to answer anything.
"""

from __future__ import annotations

import re

from chunking.base import ChunkingStrategy, approx_tokens, split_paragraphs, split_sentences
from data.loader import Passage

# Clause-level separators, used only when a single sentence still exceeds the budget.
CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


class RecursiveChunker(ChunkingStrategy):
    def __init__(
        self,
        target_tokens: int = 220,
        max_tokens: int = 320,
        min_chunk_chars: int = 80,
        name: str = "recursive",
    ) -> None:
        self.name = name
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_chunk_chars = min_chunk_chars

    # -- the recursive descent ---------------------------------------------- #

    def _split_recursive(self, text: str, depth: int = 0) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if approx_tokens(text) <= self.max_tokens:
            return [text]

        if depth == 0:
            parts = split_paragraphs(text)
        elif depth == 1:
            parts = split_sentences(text)
        elif depth == 2:
            parts = [p.strip() for p in CLAUSE_SPLIT.split(text) if p.strip()]
        else:
            # Last resort: a hard character cut. Reached only by text with no internal
            # punctuation at all, where there is no boundary left to respect.
            width = max(self.min_chunk_chars, int(self.max_tokens * len(text) / max(approx_tokens(text), 1)))
            return [text[i : i + width].strip() for i in range(0, len(text), width)]

        # No progress at this level (one part, same text) — descend rather than loop.
        if len(parts) <= 1:
            return self._split_recursive(text, depth + 1)

        out: list[str] = []
        for part in parts:
            out.extend(self._split_recursive(part, depth + 1))
        return out

    # -- repacking ----------------------------------------------------------- #

    def _pack(self, pieces: list[str]) -> list[str]:
        """Greedily merge adjacent pieces up to the target, preserving order."""
        packed: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for piece in pieces:
            n = approx_tokens(piece)
            if buf and buf_tokens + n > self.target_tokens:
                packed.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(piece)
            buf_tokens += n

        if buf:
            tail = " ".join(buf)
            # Fold a runt tail into the previous chunk rather than indexing it alone.
            if packed and len(tail) < self.min_chunk_chars:
                packed[-1] = f"{packed[-1]} {tail}"
            else:
                packed.append(tail)
        return packed

    def chunk_passage(self, passage: Passage) -> list[str]:
        text = passage.text.strip()
        if not text:
            return []
        if approx_tokens(text) <= self.max_tokens:
            return [text]
        return self._pack(self._split_recursive(text))

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "levels": ["paragraph", "sentence", "clause", "hard-cut"],
            "rationale": "cuts between units of meaning, at zero embedding cost",
        }


__all__ = ["RecursiveChunker"]
