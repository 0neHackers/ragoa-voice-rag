"""FastAPI demo server — the live link and the surface used for the demo video.

Design constraints that shaped this file:

*The index loads once, at startup.* Loading 15k vectors and building the BM25 inverted
index per request would dominate every latency number on the page. The pipeline is a
process-level singleton built through `harness.factory`, the same entrypoint the
benchmark uses, so the demo cannot drift from what was measured.

*The response exposes the whole trace.* Every guardrail verdict, every per-stage timing,
and the retrieved passages are returned, not just the answer. A voice RAG demo that shows
only its answer is indistinguishable from one that made it up, and the guardrail
behaviour is the part of this build worth seeing.

*The voice path is honest.* `/ask/voice` accepts uploaded audio and runs real STT.
`/ask/text` exists for benchmarking and for judges without a microphone, and its response
carries `transcript: null` and `voice_input: false`, so a typed question can never be
presented as a spoken one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

@asynccontextmanager
async def lifespan(_: FastAPI):
    _startup()
    yield


app = FastAPI(
    title="Voice-Enabled RAG — Team 0neHackers",
    description="HH Goa 2026 Task 2. Speak a question in Hindi, get a grounded answer.",
    version=(Path(__file__).resolve().parent.parent / "VERSION").read_text().strip(),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public read-only demo; no cookies, no credentials
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_pipeline: Any = None
_startup_error: str | None = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Pipeline not ready.")
    return _pipeline


def _startup() -> None:
    """Build the pipeline once. A failure here is recorded, not raised.

    If the index is missing the process must still start and serve `/health` explaining
    why — a container that exits on boot gives a deploy platform nothing to show and
    turns a missing-file problem into an opaque crash loop.

    Idempotent: if a pipeline is already installed this returns immediately. That keeps
    tests able to inject a stub without startup loading a 220MB model over the top of it.
    """
    global _pipeline, _startup_error
    if _pipeline is not None:
        return
    try:
        from harness.factory import build_pipeline

        index_dir = os.getenv("INDEX_DIR", "index_store")
        _pipeline = build_pipeline(index_dir, with_stt=True, warmup=True)
        print(f"Pipeline ready: {_pipeline.health()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        _startup_error = f"{type(exc).__name__}: {exc}"
        print(f"STARTUP FAILED: {_startup_error}", flush=True)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health() -> JSONResponse:
    if _pipeline is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": _startup_error})
    return JSONResponse({"ok": True, **_pipeline.health()})


@app.get("/guardrails")
def guardrails() -> dict:
    pipeline = get_pipeline()
    if pipeline.guardrails is None:
        return {"enabled": False, "guardrails": []}
    return {"enabled": True, "guardrails": pipeline.guardrails.describe()}


@app.post("/ask/text")
def ask_text(body: TextQuery) -> dict:
    """Text path. Explicitly marked as not voice input."""
    response = get_pipeline().run_text(body.query, top_k=body.top_k)
    return _serialise(response, voice_input=False)


@app.post("/ask/voice")
async def ask_voice(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
) -> dict:
    """Voice path: uploaded audio → Sarvam streaming STT → RAG → grounded answer."""
    pipeline = get_pipeline()

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio exceeds the 25MB limit.")

    import io

    from stt.sarvam_client import load_audio_file

    try:
        clip = load_audio_file(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded audio ({exc}). Send WAV, MP3, FLAC, or OGG.",
        ) from exc

    if language and pipeline.stt is not None:
        pipeline.stt.config.language_code = language

    response = pipeline.run_audio(clip, top_k=top_k)
    return _serialise(response, voice_input=True)


def _serialise(response: Any, *, voice_input: bool) -> dict:
    payload = response.model_dump(mode="json")
    payload["voice_input"] = voice_input
    payload["retrieved"] = [
        {
            "chunk_id": sc["chunk"]["chunk_id"],
            "text": sc["chunk"]["text"][:600],
            "strategy": sc["chunk"]["strategy"],
            "score": round(sc["score"], 4),
            "dense_score": round(sc["dense_score"], 4) if sc.get("dense_score") is not None else None,
            "lexical_score": round(sc["lexical_score"], 4) if sc.get("lexical_score") is not None else None,
        }
        for sc in payload.get("retrieved", [])
    ]
    return payload


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
