---
title: Voice RAG — 0neHackers
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Speak Hindi, get an answer grounded in ai4bharat/MSMARCO-XI
---

# Voice-Enabled RAG — Team 0neHackers

**HH Goa 2026, Task 2.** Ask a question out loud in Hindi. Get an answer grounded in
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — or a
refusal that tells you why.

Shanzal Firoz · Aditya Vishwakarma · Kanishka Rajput

Source: [github.com/0neHackers/ragoa-voice-rag](https://github.com/0neHackers/ragoa-voice-rag)

## What it does

```
voice ──▶ Sarvam STT ──▶ [safety] ──▶ [language] ──▶ (optional en→hi) ──▶ embed
                                                                           │
                                                           dense (FAISS) ◀─┤
                                                           BM25 (inverted) ◀┘
                                                                 │
                                                              RRF fusion
                                                                 │
                                                       [confidence gate]
                                                                 │
                    answer ◀── [groundedness] ◀── Sarvam 105b-conversations
```

- **Speak or type** a Hindi question, or upload audio
- **Translate from English** with the opt-in toggle
- **Hear it** — the question and the answer read back in Hindi
- Every guardrail verdict, per-stage timing and retrieved passage is shown, because a RAG
  demo that only shows its answer is indistinguishable from one that made it up

## Numbers

Retrieval pipeline, 50 real dataset queries: **P50 102ms · P70 106ms · P100 118ms**, 100%
inside the 200ms budget. Full pipeline including generation is ~2.8s, almost all of it the
LLM call — the two are reported separately and the split is explained in the repo README.

## Setup

This Space needs one secret:

| Secret | Purpose |
|---|---|
| `SARVAM_API_KEY` | speech-to-text, generation, translation, text-to-speech |

Add it under **Settings → Variables and secrets**. Without it the app still starts and
retrieval works, but speech-to-text returns a typed error rather than a fake transcript,
and answers drop to a clearly-labelled extractive mode.

The first build takes 15–25 minutes: it embeds all 15,449 corpus chunks into the image so
the container is ready the moment it boots, rather than serving errors while it indexes.
