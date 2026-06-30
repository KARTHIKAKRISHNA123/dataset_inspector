"""Unicode script ranges + helpers for *hint-only* language signals.

Used by the classifier and language-hint stages. Remember: a script narrows the candidate languages
but never uniquely identifies one (Devanagari is shared by 8 of our target languages). This module
deliberately produces hints, not decisions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali":    [(0x0980, 0x09FF)],
    "Gurmukhi":   [(0x0A00, 0x0A7F)],
    "Gujarati":   [(0x0A80, 0x0AFF)],
    "Oriya":      [(0x0B00, 0x0B7F)],
    "Tamil":      [(0x0B80, 0x0BFF)],
    "Telugu":     [(0x0C00, 0x0C7F)],
    "Kannada":    [(0x0C80, 0x0CFF)],
    "Malayalam":  [(0x0D00, 0x0D7F)],
    "OlChiki":    [(0x1C50, 0x1C7F)],
    "MeeteiMayek": [(0xABC0, 0xABFF), (0xAAE0, 0xAAFF)],
    "Arabic":     [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Latin":      [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}

SCRIPT_TO_LANGUAGES: Dict[str, List[str]] = {
    "Devanagari": ["hi", "mr", "ne", "kok", "mai", "doi", "brx", "sa"],
    "Bengali": ["bn", "as"],
    "Gurmukhi": ["pa"],
    "Gujarati": ["gu"],
    "Oriya": ["or"],
    "Tamil": ["ta"],
    "Telugu": ["te"],
    "Kannada": ["kn"],
    "Malayalam": ["ml"],
    "OlChiki": ["sat"],
    "MeeteiMayek": ["mni"],
    "Arabic": ["ur", "sd", "ks"],
    "Latin": ["en"],
}

# Indic scripts only (used to decide "is this an Indian-language sample" and romanization).
_INDIC_SCRIPTS = {
    "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu",
    "Kannada", "Malayalam", "OlChiki", "MeeteiMayek", "Arabic",
}


def script_of_char(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_histogram(text: str) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ch in text:
        name = script_of_char(ch)
        if name is not None:
            hist[name] = hist.get(name, 0) + 1
    return hist


def dominant_script(text: str) -> Optional[str]:
    hist = script_histogram(text)
    if not hist:
        return None
    return max(hist.items(), key=lambda kv: kv[1])[0]


def script_distribution(text: str) -> Dict[str, float]:
    """Normalised script proportions (sums to ~1 over scripted chars)."""
    hist = script_histogram(text)
    total = sum(hist.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}


def unique_language_for_script(script: str) -> Optional[str]:
    cands = SCRIPT_TO_LANGUAGES.get(script, [])
    return cands[0] if len(cands) == 1 else None


def analyze_scripts(text: str) -> Dict[str, object]:
    """Bundle the hint signals the classifier needs from one text sample.

    Returns dominant script, distribution, and booleans for mixed-script, romanized-Indic, and
    code-switching. These are heuristics:
      * mixed_language  : >=2 *Indic-or-Latin* scripts each holding a non-trivial share.
      * romanized       : Latin-only text whose path/context suggests an Indian language (caller
                          decides with path hint; here we only flag 'Latin-dominant, no Indic').
      * code_switching  : Latin present alongside an Indic script in the same sample.
    """
    dist = script_distribution(text)
    dom = next(iter(dist), None)
    indic_present = [s for s in dist if s in _INDIC_SCRIPTS and dist[s] >= 0.1]
    latin_share = dist.get("Latin", 0.0)
    mixed = len([s for s in dist if dist[s] >= 0.15]) >= 2
    code_switch = latin_share >= 0.1 and len(indic_present) >= 1
    romanized = (dom == "Latin" and not indic_present and latin_share >= 0.9)
    return {
        "dominant_script": dom,
        "script_distribution": dist,
        "mixed_language": bool(mixed),
        "romanized": bool(romanized),
        "code_switching": bool(code_switch),
        "indic_scripts_present": indic_present,
    }
