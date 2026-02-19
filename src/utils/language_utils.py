from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from langdetect import detect, DetectorFactory, LangDetectException
from lingua import Language, LanguageDetectorBuilder

# Optional INLD (Hinglish detector)
try:
    from INLD.INLD import detect as inld_detect
    INLD_AVAILABLE = True
except ImportError:
    INLD_AVAILABLE = False

# Make langdetect deterministic
DetectorFactory.seed = 0

# Languages covered by the TextDetox toxicity models (15) + Hinglish code "hin"
# en, ru, uk, de, es, am, zh, ar, hi, it, fr, he, hin, ja, tt
SUPPORTED_LANGUAGES = {
    "en", "ru", "uk", "de", "es", "am", "zh", "ar",
    "hi", "it", "fr", "he", "hin", "ja", "tt"
}

# Subset we can currently detect reliably with langdetect+lingua
ALLOWED_LANGUAGES = {
    "en", "ru", "uk", "de", "es", "ar",
    "hi", "zh", "it", "fr", "he", "ja"
}

LINGUA_LANG_MAP = {
    Language.ENGLISH: "en",
    Language.RUSSIAN: "ru",
    Language.UKRAINIAN: "uk",
    Language.GERMAN: "de",
    Language.SPANISH: "es",
    Language.ARABIC: "ar",
    Language.HINDI: "hi",
    Language.CHINESE: "zh",
    Language.ITALIAN: "it",
    Language.FRENCH: "fr",
    Language.HEBREW: "he",
    Language.JAPANESE: "ja",
}

lingua_detector = LanguageDetectorBuilder.from_languages(
    *LINGUA_LANG_MAP.keys()
).build()


@dataclass
class LanguageResult:
    lang: str          # final classification (en, ru, hin, other, ...)
    langdetect: str    # raw langdetect output
    lingua: str        # raw lingua output
    detector: str      # "agreement" | "inld" | "other"


def normalize(code: str) -> str:
    if not code:
        return "unknown"
    code = code.lower().strip()
    if code in {"zh-cn", "zh-tw"}:
        return "zh"
    if code == "iw":
        return "he"
    return code


def run_inld(text: str) -> Optional[str]:
    """
    Run INLD and return 'hin' (Hinglish) if detected.
    INLD returns ['hng'], ['hi'], ['en'], etc.
    """
    if not INLD_AVAILABLE:
        return None
    try:
        result = inld_detect(text)
        if result and result[0].lower() == "hng":
            return "hin"
    except Exception:
        pass
    return None


def detect_language(text: str) -> LanguageResult:
    if not text or not text.strip():
        return LanguageResult("other", "unknown", "unknown", "other")

    text = text.strip()

    # -------- langdetect --------
    try:
        ld_code = normalize(detect(text))
    except LangDetectException:
        ld_code = "unknown"

    # -------- lingua --------
    lingua_lang = lingua_detector.detect_language_of(text)
    if lingua_lang is None:
        li_code = "unknown"
    else:
        li_code = LINGUA_LANG_MAP.get(lingua_lang, "unknown")

    # -------- agreement rule --------
    if ld_code == li_code and ld_code in ALLOWED_LANGUAGES:
        return LanguageResult(
            lang=ld_code,
            langdetect=ld_code,
            lingua=li_code,
            detector="agreement",
        )

    # -------- INLD fallback (Hinglish) --------
    inld_lang = run_inld(text)
    if inld_lang:
        return LanguageResult(
            lang=inld_lang,
            langdetect=ld_code,
            lingua=li_code,
            detector="inld",
        )

    # -------- final fallback --------
    # For now, am + tt and low-confidence cases go here
    return LanguageResult(
        lang="other",
        langdetect=ld_code,
        lingua=li_code,
        detector="other",
    )


def detect_language_code(text: str) -> str:
    """
    Convenience helper: just returns the final language code.
    """
    return detect_language(text).lang


def detect_language_batch(texts: List[str]) -> List[LanguageResult]:
    return [detect_language(t) for t in texts]


def detect_language_codes(texts: List[str]) -> List[str]:
    return [detect_language(t).lang for t in texts]


if __name__ == "__main__":
    tests = [
        "Hello, how are you?",
        "Привет, как дела?",
        "नमस्ते, आप कैसे हैं?",
        "Hola, ¿cómo estás?",
        "こんにちは、お元気ですか？",
        "ye ek example hai",
        "Bhai aaj kya kar rahe ho?",
        "lol ok",
        ""
    ]

    for t in tests:
        r = detect_language(t)
        print(
            f"{t!r:35} → {r.lang:9} | "
            f"ld={r.langdetect:5} | li={r.lingua:5} | via={r.detector}"
        )
