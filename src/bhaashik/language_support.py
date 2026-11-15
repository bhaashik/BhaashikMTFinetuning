"""Explicit native model coverage for seven focus languages.

Corpus tags for unsupported varieties are project-local labels, not claims
that a published NLLB or IndicTrans2 checkpoint supports them.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str
    corpus_tag: str
    nllb_native: bool
    indictrans2_native: bool
    bhashaverse_card: bool
    bhashaverse_paper: bool


LANGUAGES = (
    Language("Bhojpuri", "bho_Deva", True, False, False, True),
    Language("Maithili", "mai_Deva", True, True, True, True),
    Language("Magahi", "mag_Deva", True, False, False, True),
    Language("Awadhi", "awa_Deva", True, False, False, True),
    Language("Braj", "bra_Deva", False, False, False, True),
    Language("Nepali", "npi_Deva", True, True, True, True),
    Language("Marwari", "mwr_Deva", False, False, False, False),
)
LANGUAGE_BY_TAG = {lang.corpus_tag: lang for lang in LANGUAGES}
LANGUAGE_BY_NAME = {lang.name.casefold(): lang for lang in LANGUAGES}


def language_options():
    """Return coverage metadata for notebook tables, not inferred support."""
    return [dict(name=lang.name, corpus_tag=lang.corpus_tag,
                 nllb="native" if lang.nllb_native else "no native tag",
                 indictrans2="native" if lang.indictrans2_native else "unsupported",
                 bhashaverse=("HF card" if lang.bhashaverse_card else
                              "paper only; needs checkpoint validation" if lang.bhashaverse_paper else
                              "not confirmed"))
            for lang in LANGUAGES]


def require_native_nllb(src, tgt):
    """Prevent implicit substitution of other language tags."""
    for code in (src, tgt):
        lang = LANGUAGE_BY_TAG.get(code)
        if lang is not None and not lang.nllb_native:
            raise ValueError(
                f"{lang.name} ({code}) has no native NLLB token; "
                "use an explicitly trained extension, not another language's tag")
