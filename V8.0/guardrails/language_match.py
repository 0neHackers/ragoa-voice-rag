"""Guardrail 2 — corpus-language match, pre-retrieval.

This slot originally held an embedding-based off-topic detector that scored the query
against the corpus centroid. It was implemented, measured, and removed, because the
measurement said it was **inverted**. On the 3,039-chunk Hindi index:

    query                              centroid-sim    max-sim to corpus
    "भारत की राजधानी क्या है?"  (real)      -0.035            0.440
    "मधुमेह के लक्षण क्या हैं?"  (real)       0.252            0.678
    "asdkjh qwe zxcvbn"      (gibberish)      0.184            0.381
    "aaaaa bbbbb ccccc"      (gibberish)      0.234            0.398

Real questions scored *below* gibberish. The reason is structural rather than a tuning
problem: a corpus centroid points in the "average passage" direction, so similarity to it
measures how generic a text is, not how on-topic. A specific question is nearly
orthogonal to the mean; bland token soup is not. No threshold fixes an inverted signal.

The column that does separate them is max-similarity over the whole corpus — which is
precisely the top dense score the retrieval-confidence gate already thresholds. So a
second semantic gate would not be an independent check, just a less precise copy of one
the pipeline already runs. **Semantic off-topic detection is handled by
`low_confidence`**, and it is the right place for it: after retrieval, with the actual
best score in hand rather than a proxy for it.

What is left over is a failure the confidence gate provably *cannot* catch. In the same
measurement, the English query "what is the capital of france" scored **0.628** against
the Hindi corpus — comfortably above the 0.42 confidence threshold — because a
multilingual embedding model maps it near Hindi passages about countries and capitals,
and MS MARCO, being open-domain web text, really does contain such passages. Left alone,
that query gets a confidently generated answer synthesised from Hindi passages that were
never about it. This guardrail catches it in ~0.02ms without embedding anything.

Code-mixing is handled explicitly, because Sarvam transcribes code-mixed speech and a
Hinglish transcript is *not* a language mismatch. A Latin-script query carrying romanised
Hindi function words ("bharat ki rajdhani kya hai") is exempted; it is a real Hindi
question that happens to be romanised, and the confidence gate can judge it on merit.
"""

from __future__ import annotations

import re
import unicodedata

from harness.types import GuardrailVerdict
from guardrails.base import Guardrail

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

#: Romanised Hindi function words. Their presence marks a Latin-script query as Hinglish
#: rather than English. Function words specifically — they appear in essentially every
#: Hindi sentence regardless of topic, which content words do not.
#:
#: Words that are *also* ordinary English words are excluded, even though they are
#: perfectly good romanised Hindi. `the` (थे, "were") is the clearest case: including it
#: made "what is the capital of france" exempt itself as Hinglish and sail through the
#: guardrail. Also excluded for the same reason: `me` (में — `mein` is kept, and is the
#: more common romanisation), `par` (पर), `to` (तो), `is` (इस), `us` (उस), `bat` (बात).
#: A marker that fires on English costs the guardrail its entire purpose, while a missing
#: marker costs only a little recall on Hinglish — the asymmetry decides the list.
ROMANISED_HINDI_MARKERS: frozenset[str] = frozenset("""
kya kyu kyun kaise kaisa kaisi kab kahan kaun kitna kitne kitni
hai hain tha thi hota hoti hote hona
ka ki ke ko se mein aur ya nahi nahin
mera meri mere tera teri tumhara aapka apna apne
bata batao bataiye samjhao karo karna kiya
""".split())

#: Below this Devanagari fraction, a corpus is not treated as Devanagari and the
#: guardrail disables itself rather than guessing.
CORPUS_SCRIPT_THRESHOLD = 0.5

#: A query must be at least this Latin-dominant to be considered a mismatch. Leaves room
#: for the stray English technical term inside an otherwise Hindi question.
QUERY_LATIN_THRESHOLD = 0.8


class LanguageMatchGuardrail(Guardrail):
    name = "language_mismatch"
    fail_open = True

    def __init__(self, corpus_script: str = "devanagari", enabled: bool = True) -> None:
        self.corpus_script = corpus_script
        self.enabled = enabled

    @classmethod
    def from_chunks(cls, chunks: list, sample: int = 400) -> "LanguageMatchGuardrail":
        """Infer the corpus script from the corpus itself, rather than trusting config.

        `DATASET_LANG` says what was *requested*; this measures what was actually
        indexed. They diverge whenever someone rebuilds with a different subset and
        forgets to update `.env`, and a guardrail keyed to the stale value would then
        reject every legitimate query.
        """
        texts = [c.text for c in chunks[:sample]]
        if not texts:
            return cls(enabled=False)

        blob = " ".join(texts)
        deva = len(_DEVANAGARI.findall(blob))
        latin = len(_LATIN.findall(blob))
        total = deva + latin
        if total == 0:
            return cls(enabled=False)

        if deva / total >= CORPUS_SCRIPT_THRESHOLD:
            return cls(corpus_script="devanagari", enabled=True)
        return cls(corpus_script="latin", enabled=False)

    def _check(self, query: str, query_vector=None) -> GuardrailVerdict:  # type: ignore[override]
        if not self.enabled or self.corpus_script != "devanagari":
            return self._pass(reason="Corpus is not script-restricted; check disabled.")

        text = unicodedata.normalize("NFKC", query or "")
        deva = len(_DEVANAGARI.findall(text))
        latin = len(_LATIN.findall(text))
        total = deva + latin

        if total == 0:
            return self._pass(reason="Query carries no script evidence to judge.")

        latin_fraction = latin / total
        if latin_fraction < QUERY_LATIN_THRESHOLD:
            return self._pass(score=latin_fraction, threshold=QUERY_LATIN_THRESHOLD,
                              reason="Query is in the corpus's script.")

        words = {w.lower() for w in _WORD.findall(text)}
        markers = words & ROMANISED_HINDI_MARKERS
        if markers:
            return self._pass(
                score=latin_fraction, threshold=QUERY_LATIN_THRESHOLD,
                reason=f"Romanised Hindi detected ({', '.join(sorted(markers)[:3])}); not a mismatch.",
            )

        return self._block(
            reason=(
                f"Query is {latin_fraction:.0%} Latin script with no romanised-Hindi markers, "
                "but this corpus is Hindi (Devanagari). Retrieval would return passages that "
                "are topically adjacent but not about this question."
            ),
            score=latin_fraction, threshold=QUERY_LATIN_THRESHOLD,
        )


__all__ = ["LanguageMatchGuardrail", "ROMANISED_HINDI_MARKERS", "QUERY_LATIN_THRESHOLD"]
