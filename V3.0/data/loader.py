"""Loading and normalising `ai4bharat/MSMARCO-XI`.

Two things about this dataset drive the design here.

First, it is **not** exposed as named language configs — `load_dataset(..., "hi")` fails.
The repo is a flat set of per-language parquet files (`validation/hinval.parquet`), so
we address the file directly.

Second, the Hindi train parquet is 3.7GB as a single file, which is a bad trade for a
corpus of a few thousand examples on a hackathon timeline. We read the validation
parquet (462MB, identical schema) and cache an extracted slice locally as JSONL, so the
download is paid once and every subsequent index build reads from disk in milliseconds.

Row schema (observed, not assumed from MS MARCO docs):
    source_lang, target_lang, meta, Answer, query_id, query_type,
    passages, Eng_Query, Eng_Answer, query
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

HF_REPO = "ai4bharat/MSMARCO-XI"

# Language code -> the parquet basename used inside the repo.
LANG_FILES: dict[str, str] = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


@dataclass(slots=True)
class Passage:
    """One retrievable passage plus the metadata metadata-aware chunking needs."""

    text: str
    query_id: str
    passage_idx: int
    source_lang: str
    is_selected: bool = False
    url: str | None = None
    english_text: str | None = None


@dataclass(slots=True)
class Example:
    """One dataset row: a question, its reference answer, and its candidate passages."""

    query_id: str
    query: str
    answer: str
    query_type: str
    source_lang: str
    eng_query: str = ""
    passages: list[Passage] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        a = self.answer.strip().lower()
        return bool(a) and a not in ("no answer present.", "no answer present")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def _coerce_passages(
    raw: Any, query_id: str, lang: str, prefer_translated: bool = True
) -> list[Passage]:
    """Normalise the `passages` column into `Passage` objects.

    The real MSMARCO-XI shape (confirmed against `validation/hinval.parquet`) is a
    struct of three parallel lists: `Translated_passages` (the target-language text —
    what we retrieve over), `English_passages` (the aligned source text), and
    `is_selected` (int 0/1 answer labels). Roughly 10 passages per query.

    The English column is carried alongside rather than dropped: it costs nothing at
    load time and makes an English-query demo possible without a second download.

    Fallbacks for the plain MS MARCO shapes are kept because this loader is also aimed
    at the other 13 language files, and nothing guarantees they were all written by the
    same export script.
    """
    out: list[Passage] = []
    if raw is None:
        return out

    # Shape A: struct of parallel lists (the MSMARCO-XI shape)
    if isinstance(raw, dict):
        translated = raw.get("Translated_passages") or []
        english = raw.get("English_passages") or []
        primary = (translated or english) if prefer_translated else (english or translated)
        alternate = english if primary is translated else translated

        # Plain MS MARCO fallbacks
        if not primary:
            primary = raw.get("passage_text") or raw.get("text") or raw.get("passage") or []
            alternate = []

        selected = raw.get("is_selected") or []
        urls = raw.get("url") or []
        for i, text in enumerate(primary):
            if not isinstance(text, str) or not text.strip():
                continue
            out.append(Passage(
                text=text.strip(),
                query_id=query_id,
                passage_idx=i,
                source_lang=lang,
                # is_selected arrives as int 0/1, not bool
                is_selected=bool(selected[i]) if i < len(selected) else False,
                url=urls[i] if i < len(urls) else None,
                english_text=(alternate[i].strip() if i < len(alternate)
                              and isinstance(alternate[i], str) else None),
            ))
        return out

    # Shape B: list of structs (or of bare strings)
    if isinstance(raw, (list, tuple)):
        for i, item in enumerate(raw):
            if isinstance(item, str):
                text, sel, url = item, False, None
            elif isinstance(item, dict):
                text = item.get("passage_text") or item.get("text") or item.get("passage") or ""
                sel = bool(item.get("is_selected", False))
                url = item.get("url")
            else:
                continue
            if isinstance(text, str) and text.strip():
                out.append(Passage(text=text.strip(), query_id=query_id, passage_idx=i,
                                   source_lang=lang, is_selected=sel, url=url))
    return out


def _row_to_example(row: dict[str, Any], lang: str) -> Example | None:
    query = (row.get("query") or "").strip()
    if not query:
        return None
    qid = str(row.get("query_id") or "")
    return Example(
        query_id=qid,
        query=query,
        answer=(row.get("Answer") or row.get("answer") or "").strip(),
        query_type=(row.get("query_type") or "unknown"),
        source_lang=row.get("target_lang") or lang,
        eng_query=(row.get("Eng_Query") or "").strip().lstrip(". ").strip(),
        passages=_coerce_passages(row.get("passages"), qid, lang),
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _cache_path(lang: str, split: str, cache_dir: Path) -> Path:
    """One cache file per (lang, split) — deliberately not per-limit.

    Keying the cache on `limit` too would mean every new corpus size re-reads the
    parquet, and that read is the single slowest step in the whole build (~5 min for
    the 462MB Hindi file, which stores all 97,941 rows in one row group, so there is no
    partial-read shortcut). Caching a generous slice once and taking a prefix makes
    every subsequent size change instant.
    """
    return cache_dir / f"msmarco_xi_{lang}_{split}.jsonl"


#: How much to extract on a cache miss, regardless of the limit asked for.
CACHE_EXTRACT_SIZE = 5000


def load_examples(
    lang: str = "hi",
    split: str = "validation",
    limit: int = 2000,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> list[Example]:
    """Return up to `limit` normalised examples, using a local JSONL cache when present."""
    cache_dir = Path(cache_dir or Path(__file__).resolve().parent.parent / "corpus_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(lang, split, cache_dir)

    if cache.exists() and not force_refresh:
        cached = _read_cache(cache, limit)
        # Only re-extract if the cache genuinely cannot satisfy the request — i.e. it
        # holds fewer than asked *and* fewer than we would have extracted anyway.
        if len(cached) >= limit or len(cached) >= CACHE_EXTRACT_SIZE:
            return cached[:limit]

    extract = max(limit, CACHE_EXTRACT_SIZE)
    examples = list(_read_parquet(lang, split, extract))
    _write_cache(cache, examples)
    return examples[:limit]


def _read_parquet(lang: str, split: str, limit: int) -> Iterator[Example]:
    """Stream row groups off the hub parquet, stopping as soon as `limit` is reached."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if lang not in LANG_FILES:
        raise ValueError(f"Unsupported language {lang!r}. Available: {sorted(LANG_FILES)}")
    if split not in ("train", "validation"):
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")

    suffix = "train" if split == "train" else "val"
    filename = f"{split}/{LANG_FILES[lang]}{suffix}.parquet"

    os.environ.setdefault(
        "HF_HOME",
        str(Path(__file__).resolve().parent.parent.parent / "hf_cache"),
    )
    path = hf_hub_download(HF_REPO, filename, repo_type="dataset")

    pf = pq.ParquetFile(path)
    seen = 0
    # Row-group-at-a-time rather than `read().to_pylist()`: the train files are multi-GB
    # and materialising the whole table to build a 2000-example corpus is pure waste.
    for batch in pf.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            ex = _row_to_example(row, lang)
            if ex is None or not ex.passages:
                continue
            yield ex
            seen += 1
            if seen >= limit:
                return


def _write_cache(path: Path, examples: list[Example]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({
                "query_id": ex.query_id, "query": ex.query, "answer": ex.answer,
                "query_type": ex.query_type, "source_lang": ex.source_lang,
                "eng_query": ex.eng_query,
                "passages": [
                    {"text": p.text, "passage_idx": p.passage_idx,
                     "is_selected": p.is_selected, "url": p.url,
                     "english_text": p.english_text}
                    for p in ex.passages
                ],
            }, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: a half-written cache must never be readable


def _read_cache(path: Path, limit: int | None = None) -> list[Example]:
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(out) >= limit:
                break
            d = json.loads(line)
            out.append(Example(
                query_id=d["query_id"], query=d["query"], answer=d["answer"],
                query_type=d["query_type"], source_lang=d["source_lang"],
                eng_query=d.get("eng_query", ""),
                passages=[
                    Passage(text=p["text"], query_id=d["query_id"],
                            passage_idx=p["passage_idx"], source_lang=d["source_lang"],
                            is_selected=p["is_selected"], url=p.get("url"),
                            english_text=p.get("english_text"))
                    for p in d["passages"]
                ],
            ))
    return out


def corpus_stats(examples: list[Example]) -> dict[str, Any]:
    passages = [p for ex in examples for p in ex.passages]
    lengths = [len(p.text) for p in passages]
    return {
        "examples": len(examples),
        "passages": len(passages),
        "answerable_examples": sum(ex.has_answer for ex in examples),
        "passages_per_example": round(len(passages) / max(len(examples), 1), 2),
        "mean_passage_chars": round(sum(lengths) / max(len(lengths), 1), 1),
        "max_passage_chars": max(lengths, default=0),
        "selected_passages": sum(p.is_selected for p in passages),
    }


__all__ = ["Example", "Passage", "load_examples", "corpus_stats", "LANG_FILES"]
