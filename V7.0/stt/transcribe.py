"""Standalone speech-to-text entrypoint — Phase 1 deliverable.

Real audio in, transcript out. Nothing downstream of STT is involved.

    python -m stt.transcribe --mic --seconds 6
    python -m stt.transcribe --file audio_samples/question_hi.wav
    python -m stt.transcribe --list-devices
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python stt/transcribe.py` as well as `python -m stt.transcribe`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from stt.sarvam_client import (  # noqa: E402
    Audio, SarvamConfig, SarvamSTT, load_audio_file, record_microphone,
)


def list_devices() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        print(f"sounddevice unavailable: {exc}", file=sys.stderr)
        return 2
    print(sd.query_devices())
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Transcribe real audio via Sarvam AI.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mic", action="store_true", help="record from the microphone")
    src.add_argument("--file", type=str, help="path to a .wav/.mp3/.flac question")
    ap.add_argument("--seconds", type=float, default=6.0, help="mic recording length")
    ap.add_argument("--language", type=str, default=None, help="e.g. hi-IN, en-IN")
    ap.add_argument("--save", type=str, default=None, help="save captured mic audio to this .wav")
    ap.add_argument("--no-fallback", action="store_true",
                    help="fail instead of falling back to the batch endpoint")
    ap.add_argument("--json", action="store_true", help="emit the structured result as JSON")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args(argv)

    if args.list_devices:
        return list_devices()

    if not args.mic and not args.file:
        ap.error("one of --mic or --file is required (this stage does not accept typed text)")

    # ---- acquire real audio -------------------------------------------------
    try:
        if args.mic:
            print(f"Recording {args.seconds:.0f}s — speak your question now...", file=sys.stderr)
            audio: Audio = record_microphone(args.seconds)
            print("Recording finished.", file=sys.stderr)
            if args.save:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                Path(args.save).write_bytes(audio.to_wav_bytes())
                print(f"Saved audio -> {args.save}", file=sys.stderr)
        else:
            path = Path(args.file)
            if not path.exists():
                print(f"No such audio file: {path}", file=sys.stderr)
                return 2
            audio = load_audio_file(path)
    except Exception as exc:  # noqa: BLE001
        print(f"Audio capture failed: {exc}", file=sys.stderr)
        return 2

    print(f"Audio: {audio.duration_s:.2f}s @ {audio.sample_rate}Hz mono", file=sys.stderr)

    # ---- transcribe ---------------------------------------------------------
    cfg = SarvamConfig.from_env()
    if args.language:
        cfg.language_code = args.language

    result = SarvamSTT(cfg).transcribe(audio, allow_batch_fallback=not args.no_fallback)

    if args.json:
        payload = {
            "ok": result.ok,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "transcript": result.value.model_dump() if result.value else None,
            "error": result.error.model_dump() if result.error else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.ok else 1

    if not result.ok:
        print(f"\nFAILED  {result.error}", file=sys.stderr)
        return 1

    t = result.value
    print(f"\nTranscript : {t.text}")
    print(f"Language   : {t.language_code}")
    print(f"Model      : {t.provider}/{t.model} via {t.transport}")
    print(f"Latency    : {result.elapsed_ms:.0f}ms for {t.audio_duration_s:.2f}s of audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
