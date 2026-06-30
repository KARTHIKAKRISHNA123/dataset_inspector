"""Language hints — path-based and script-based. Hints only, never a final decision.

The inspector's job ends at *hints*: it must not commit to a language, because (a) shared scripts are
ambiguous and (b) final language-ID belongs to a dedicated model in the downstream pipeline. We
expose two cheap, explainable signals and a confidence that reflects how much they agree.
"""

from __future__ import annotations

from typing import Optional

from ..core.models import LanguageHints
from ..core.scripts import analyze_scripts, dominant_script, unique_language_for_script

# Path-token → ISO-639 code. Longest match wins.
_PATH_HINTS = {
    "assam": "as", "bengali": "bn", "bangla": "bn", "bodo": "brx", "dogri": "doi",
    "gujarat": "gu", "hindi": "hi", "kannada": "kn", "kashmir": "ks", "konkani": "kok",
    "maithili": "mai", "malayalam": "ml", "manipuri": "mni", "meitei": "mni", "marathi": "mr",
    "nepali": "ne", "odia": "or", "oriya": "or", "punjabi": "pa", "panjabi": "pa",
    "sanskrit": "sa", "santhali": "sat", "santali": "sat", "sindhi": "sd", "tamil": "ta",
    "telugu": "te", "urdu": "ur", "english": "en",
    # common ISO suffix tokens
    "_as": "as", "_bn": "bn", "_gu": "gu", "_hi": "hi", "_kn": "kn", "_ml": "ml",
    "_mr": "mr", "_ne": "ne", "_or": "or", "_pa": "pa", "_sa": "sa", "_ta": "ta",
    "_te": "te", "_ur": "ur", "_en": "en",
}


def path_language_hint(path: str, archive_path: Optional[str] = None) -> Optional[str]:
    hay = ((archive_path or "") + "/" + path).lower().replace("-", "_").replace(" ", "_")
    for frag in sorted(_PATH_HINTS, key=len, reverse=True):
        if frag in hay:
            return _PATH_HINTS[frag]
    return None


def language_hints_from_text(text: str, path_hint: Optional[str] = None) -> LanguageHints:
    """Combine script analysis of a text sample with an optional path hint into LanguageHints.

    Confidence logic (transparent, tunable):
      * unique script (e.g. Tamil) → 0.9
      * shared script but path hint agrees with script's candidate set → 0.75
      * path hint only (no scripted text / romanized) → 0.5
      * shared script, no path hint → 0.4
      * nothing → 0.0
    """
    info = analyze_scripts(text or "")
    dom = info["dominant_script"]
    script_lang = unique_language_for_script(dom) if dom else None

    conf = 0.0
    script_hint = script_lang
    if script_lang:
        conf = 0.9
    elif dom:  # shared script
        conf = 0.4
        if path_hint:
            conf = 0.75
    if info["romanized"] and path_hint:
        conf = max(conf, 0.5)
    if not dom and path_hint:
        conf = max(conf, 0.5)

    return LanguageHints(
        path_hint=path_hint,
        script_hint=script_hint,
        dominant_script=dom,
        script_distribution=info["script_distribution"],   # type: ignore[arg-type]
        mixed_language=bool(info["mixed_language"]),
        romanized=bool(info["romanized"]),
        code_switching=bool(info["code_switching"]),
        confidence=round(conf, 2),
    )
