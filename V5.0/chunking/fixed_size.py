"""Strategy 1 — fixed-size chunking with overlap. The baseline.

Kept simple, and kept around: it's the control the other three
strategies are measured against. Its weakness is the reason the others exist — it cuts
mid-sentence and mid-idea, so a chunk can end up holding the first half of a claim and
the second half of an unrelated one.

Overlap is the mitigation: a fact straddling a boundary still appears intact in one of
the two neighbouring chunks. It costs index size in exchange for recall.
"""

from __future__ import annotations

from chunking.base import ChunkingStrategy, approx_tokens
from data.loader import Passage


class FixedSizeChunker(ChunkingStrategy):
    """Character-window chunker with a configurable overlap ratio.

    Sizing is in characters rather than tokens because the window is applied by slicing
    and a token-exact window would require a tokenizer round trip per passage for no
    retrieval benefit. `target_tokens` is converted to characters using the script-aware
    ratio in `chunking.base.approx_tokens`.
    """

    def __init__(
        self,
        target_tokens: int = 256,
        overlap_ratio: float = 0.18,
        min_chunk_chars: int = 80,
        name: str = "fixed_size",
    ) -> None:
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0.0, 1.0)")
        self.name = name
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.min_chunk_chars = min_chunk_chars

    def _window_chars(self, text: str) -> int:
        """Characters per window, derived from the target token count for this script."""
        tokens = approx_tokens(text)
        if tokens == 0:
            return self.target_tokens * 4
        chars_per_token = len(text) / tokens
        return max(self.min_chunk_chars, int(self.target_tokens * chars_per_token))

    def chunk_passage(self, passage: Passage) -> list[str]:
        text = passage.text.strip()
        if not text:
            return []

        window = self._window_chars(text)
        if len(text) <= window:
            return [text]

        stride = max(1, int(window * (1.0 - self.overlap_ratio)))
        chunks: list[str] = []
        for start in range(0, len(text), stride):
            piece = text[start : start + window].strip()
            if not piece:
                continue
            # A trailing sliver shorter than the floor is overlap-covered by the
            # previous chunk already; emitting it only adds a low-information vector.
            if len(piece) < self.min_chunk_chars and chunks:
                break
            chunks.append(piece)
            if start + window >= len(text):
                break
        return chunks

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "target_tokens": self.target_tokens,
            "overlap_ratio": self.overlap_ratio,
            "rationale": "baseline; overlap preserves facts that straddle a cut",
        }


__all__ = ["FixedSizeChunker"]
