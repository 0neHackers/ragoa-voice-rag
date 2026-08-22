"""Single place that assembles a live pipeline from a built index.

Every entrypoint — the demo server, the benchmark, the CLI, the tests — builds its
pipeline here, so they cannot drift into measuring subtly different systems. A benchmark
that constructs its own retriever with its own defaults is a benchmark of something other
than what gets deployed.

The embedder is warmed before the pipeline is returned. The first ONNX inference pays a
one-off graph-initialisation cost of several hundred milliseconds; leaving that in means
the first real query — the one in the demo video — is the slowest one the system ever
serves, and a benchmark's P100 measures cold start rather than the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_INDEX_DIR = "index_store"


def build_pipeline(
    index_dir: str | Path | None = None,
    *,
    with_stt: bool = True,
    with_generation: bool = True,
    with_guardrails: bool = True,
    with_translation: bool = True,
    warmup: bool = True,
) -> Any:
    """Load the index and return a ready `VoiceRAGPipeline`."""
    load_dotenv()

    from generation.generator import Generator
    from guardrails.suite import GuardrailSuite
    from harness.orchestrator import PipelineConfig, VoiceRAGPipeline
    from retrieval.lexical import LexicalIndex
    from retrieval.retriever import Retriever
    from retrieval.vector_index import VectorIndex

    directory = Path(index_dir or os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR))
    if not directory.exists():
        raise FileNotFoundError(
            f"No index at {directory.resolve()}. Build one first:\n"
            f"    python -m retrieval.build_index --limit 1500 --out {directory}"
        )

    vector_index = VectorIndex.load(directory)
    lexical_index = LexicalIndex(vector_index.chunks)
    retriever = Retriever(vector_index=vector_index, lexical_index=lexical_index)

    if warmup:
        retriever.embedder.warmup()

    stt = None
    if with_stt:
        from stt.sarvam_client import SarvamSTT

        stt = SarvamSTT()

    config = PipelineConfig.from_env()
    config.enable_generation = with_generation and config.enable_generation
    config.enable_guardrails = with_guardrails and config.enable_guardrails

    translator = None
    if with_translation:
        from translation.translator import Translator

        translator = Translator()

    return VoiceRAGPipeline(
        retriever=retriever,
        stt=stt,
        translator=translator,
        generator=Generator() if config.enable_generation else None,
        guardrails=GuardrailSuite.from_retriever(retriever) if config.enable_guardrails else None,
        config=config,
    )


__all__ = ["build_pipeline", "DEFAULT_INDEX_DIR"]
