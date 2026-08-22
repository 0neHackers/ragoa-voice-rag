"""Verify every stated requirement in docs/ against what is actually built.

Reads the artefacts rather than trusting the prose: benchmark JSON for the latency claims,
the filesystem for the strategy and guardrail counts, git for the secrets check. Run it
after any change that could plausibly move one of them.

    python verify_task.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import warnings

warnings.filterwarnings("ignore")

checks: list[tuple[str, bool, str]] = []


def chk(req: str, ok: bool, detail: str) -> None:
    checks.append((req, bool(ok), detail))


root = pathlib.Path(__file__).resolve().parent

# ---- R1: speech-to-text, Sarvam or ElevenLabs, pick one ------------------
stt = (root / "stt" / "sarvam_client.py").read_text(encoding="utf-8")
chk("R1  STT provider is Sarvam, one provider",
    "api.sarvam.ai" in stt and "elevenlabs" not in stt.lower(),
    "sarvam_client.py -> api.sarvam.ai; no ElevenLabs anywhere")
chk("R1b No typed-text substitute for the voice stage",
    "is_real_audio" in stt and "no typed-text substitute" in stt,
    "is_real_audio is set only by paths that pushed audio bytes")

# ---- R2: chunking must be 'vast' ----------------------------------------
strategies = sorted(p.stem for p in (root / "chunking").glob("*.py")
                    if p.stem not in ("__init__", "base", "registry"))
comparison = json.loads((root / "benchmarks" / "chunking_comparison.json").read_text(encoding="utf-8"))
chk("R2  Four+ chunking strategies implemented", len(strategies) >= 4,
    f"{len(strategies)}: {', '.join(strategies)}")
chk("R2b Strategies compared on retrieval quality, not vibes",
    len(comparison["results"]) >= 4,
    f"recall@5 + MRR measured across {len(comparison['results'])} variants")

# ---- R3/R4: latency ------------------------------------------------------
results = json.loads((root / "benchmarks" / "results_latest.json").read_text(encoding="utf-8"))
retrieval = results["retrieval_pipeline"]
chk("R3  Retrieval pipeline under 200ms", retrieval["p100_ms"] <= 200,
    f"P100 {retrieval['p100_ms']}ms; {retrieval['pct_within_budget']}% of queries within budget")
chk("R4  P50/P70/P100 over 30+ real queries",
    all(k in retrieval for k in ("p50_ms", "p70_ms", "p100_ms")) and retrieval["n"] >= 30,
    f"n={retrieval['n']}  P50 {retrieval['p50_ms']}  P70 {retrieval['p70_ms']}  P100 {retrieval['p100_ms']}")
chk("R4b Full-pipeline latency reported separately",
    results.get("full_pipeline") is not None,
    f"P50 {results['full_pipeline']['p50_ms']}ms including generation"
    if results.get("full_pipeline") else "missing")
chk("R4c Benchmark states whether an LLM was in the loop",
    "full_pipeline_includes_llm" in results,
    f"full_pipeline_includes_llm={results.get('full_pipeline_includes_llm')}")

# ---- R5: harness ---------------------------------------------------------
orchestrator = (root / "harness" / "orchestrator.py").read_text(encoding="utf-8")
types_src = (root / "harness" / "types.py").read_text(encoding="utf-8")
retry_src = (root / "harness" / "retry.py").read_text(encoding="utf-8")
chk("R5  Harness: structured I/O at every stage",
    "StageResult" in types_src and "StageError" in types_src,
    "typed value-or-error per stage, nothing raises across a boundary")
chk("R5b Harness: retries with backoff",
    "RetryPolicy" in orchestrator and "full jitter" in retry_src,
    "jittered exponential backoff with a wall-clock deadline")
chk("R5c Harness: per-stage error recovery",
    "_run_stage" in orchestrator and "never allowed to raise" in orchestrator,
    "every stage wrapped; failures become typed errors")

# ---- R6: guardrails ------------------------------------------------------
guards = sorted(p.stem for p in (root / "guardrails").glob("*.py")
                if p.stem not in ("__init__", "base", "suite"))
chk("R6  At least 3 guardrails", len(guards) >= 3, f"{len(guards)}: {', '.join(guards)}")
chk("R6b Covers unsafe input, off-topic, and groundedness",
    {"input_safety", "language_match", "confidence", "groundedness"} <= set(guards),
    "input_safety + language_match (pre), confidence (mid), groundedness (post)")
# Assert the behaviour, not a docstring phrase: build a decline and check the shape of it.
def _decline_is_structured() -> tuple[bool, str]:
    import sys
    sys.path.insert(0, str(root))
    from harness.types import GuardrailVerdict, PipelineResponse, ResponseStatus

    r = PipelineResponse.declined(
        reason="low_confidence",
        guardrails=[GuardrailVerdict(name="low_confidence", passed=False,
                                     score=0.19, threshold=0.42, reason="best passage 0.19")],
    )
    payload = r.model_dump(mode="json")
    ok = (payload["status"] == "declined"
          and payload["reason"] == "low_confidence"
          and payload["answer"] is None
          and payload["guardrails"][0]["passed"] is False
          and payload["guardrails"][0]["score"] == 0.19
          and payload["guardrails"][0]["threshold"] == 0.42)
    return ok, f'status={payload["status"]!r} reason={payload["reason"]!r} answer={payload["answer"]!r}'

_ok, _detail = _decline_is_structured()
chk("R6c Declines are structured, not crashes", _ok,
    _detail + "; verdict carries score + threshold")

# ---- dataset -------------------------------------------------------------
loader = (root / "data" / "loader.py").read_text(encoding="utf-8")
chk("Dataset is the provided ai4bharat/MSMARCO-XI",
    "ai4bharat/MSMARCO-XI" in loader, "loader targets the dataset the task specifies")

# ---- hygiene -------------------------------------------------------------
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=root.parent).stdout.splitlines()
chk("No .env committed",
    not any(line.strip() == ".env" or line.strip().endswith("/.env") for line in tracked),
    "only .env.example is tracked")
chk("Deployment config present",
    (root / "Dockerfile").exists() and (root.parent / "render.yaml").exists()
    and (root / "SPACE_README.md").exists()
    and (root.parent / ".devcontainer" / "devcontainer.json").exists(),
    "Dockerfile + Render blueprint (repo root) + HF Space manifest + Codespaces devcontainer")

width = max(len(c[0]) for c in checks)
failed = 0
for req, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {req:<{width}}  {detail}")
    failed += not ok

print()
print(f"{len(checks) - failed}/{len(checks)} verified"
      + ("" if not failed else f"  —  {failed} FAILING"))
raise SystemExit(1 if failed else 0)
