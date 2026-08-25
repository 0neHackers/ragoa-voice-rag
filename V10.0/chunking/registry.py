"""Registry of chunking strategies — the swap point for the index builder and benchmark.

Strategies are constructed lazily (a `SemanticChunker` would otherwise load an embedding
model at import time) and named consistently with the `Chunk.strategy` tag, so a chunk
pulled out of the index can always be traced back to the strategy that produced it.
"""

from __future__ import annotations

from typing import Callable

from chunking.base import ChunkingStrategy
from chunking.fixed_size import FixedSizeChunker
from chunking.metadata_aware import MetadataAwareChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker

STRATEGIES: dict[str, Callable[..., ChunkingStrategy]] = {
    "fixed_size": FixedSizeChunker,
    "semantic": SemanticChunker,
    "recursive": RecursiveChunker,
    "metadata_aware": MetadataAwareChunker,
}

# What the index is built with unless overridden. Metadata-aware leads because the
# corpus is pre-segmented (see chunking/metadata_aware.py); recursive is the general
# fallback for any passage long enough to need real splitting.
DEFAULT_ENSEMBLE: tuple[str, ...] = ("metadata_aware", "recursive")


def get_strategy(name: str, **kwargs) -> ChunkingStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown chunking strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)


def all_strategies(**kwargs) -> list[ChunkingStrategy]:
    return [get_strategy(n, **kwargs) for n in STRATEGIES]


__all__ = ["STRATEGIES", "DEFAULT_ENSEMBLE", "get_strategy", "all_strategies",
           "FixedSizeChunker", "SemanticChunker", "RecursiveChunker", "MetadataAwareChunker"]
