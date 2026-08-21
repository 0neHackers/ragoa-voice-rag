"""Strategy 4 — metadata-aware chunking on the corpus's natural boundaries.

MSMARCO-XI passages arrive already segmented: a human-curated retrieval unit per row,
each tagged with `query_id`, `passage_idx`, `source_lang`, and `is_selected` (whether
annotators marked it as the passage containing the answer). Re-splitting those is
actively destructive — it discards editorial segmentation and replaces it with a guess.

So this strategy treats each passage as one chunk, splitting *only* when a passage
exceeds the embedding model's useful context, and it propagates the metadata into the
index so retrieval can filter and boost rather than relying on vector similarity alone.

`is_selected` is deliberately **not** used to boost at query time. It is ground-truth
answer labelling, and letting it influence ranking would leak the answer key into
retrieval and inflate every benchmark number this project reports. It is carried for
evaluation only — measuring whether retrieval independently found the right passage.
"""

from __future__ import annotations

from chunking.base import ChunkingStrategy, approx_tokens, split_sentences
from data.loader import Passage


class MetadataAwareChunker(ChunkingStrategy):
    def __init__(
        self,
        max_tokens: int = 384,
        min_chunk_chars: int = 40,
        prepend_query_context: bool = False,
        name: str = "metadata_aware",
    ) -> None:
        self.name = name
        self.max_tokens = max_tokens
        self.min_chunk_chars = min_chunk_chars
        self.prepend_query_context = prepend_query_context

    def chunk_passage(self, passage: Passage) -> list[str]:
        text = passage.text.strip()
        if not text or len(text) < self.min_chunk_chars:
            return []

        if approx_tokens(text) <= self.max_tokens:
            return [text]

        # Oversized: split on sentence boundaries into the fewest pieces that fit,
        # rather than a character window — the passage is still an editorial unit and
        # deserves the least destructive cut available.
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            width = int(self.max_tokens * len(text) / max(approx_tokens(text), 1))
            return [text[i : i + width].strip() for i in range(0, len(text), width)]

        out: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for sent in sentences:
            n = approx_tokens(sent)
            if buf and buf_tokens + n > self.max_tokens:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(sent)
            buf_tokens += n
        if buf:
            out.append(" ".join(buf))
        return [c for c in out if c.strip()]

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "max_tokens": self.max_tokens,
            "metadata_fields": ["query_id", "passage_idx", "source_lang", "is_selected_passage"],
            "rationale": "respects the corpus's own editorial segmentation; metadata enables filtered retrieval",
            "note": "is_selected is carried for evaluation only, never used to boost ranking",
        }


__all__ = ["MetadataAwareChunker"]
