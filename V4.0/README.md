# Voice-Enabled RAG — Team 0neHackers (HH Goa 2026, Task 2)

> **V0.0 — scaffold only.** Directory structure and versioning system in place; no functional
> code yet. See `../CHANGELOG.md` for version history and `DECISIONS.md` for the design calls
> that gate the build.

Pipeline shape: `voice → Sarvam streaming STT → guardrails → hybrid retrieval (FAISS + BM25)
→ guardrails → grounded generation → guardrails → structured answer`.
