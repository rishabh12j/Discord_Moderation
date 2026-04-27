from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from langdetect import detect, DetectorFactory, LangDetectException
from lingua import Language, LanguageDetectorBuilder

try:
    from INLD.INLD import detect as inld_detect
    INLD_AVAILABLE = True
except ImportError:
    INLD_AVAILABLE = False

DetectorFactory.seed = 0

ALLOWED_LANGUAGES = {
    "en", "ru", "uk", "de", "es", "ar",
    "hi", "zh", "it", "fr", "he", "ja", "hin"
}

LINGUA_LANG_MAP = {
    Language.ENGLISH: "en", Language.RUSSIAN: "ru", Language.UKRAINIAN: "uk",
    Language.GERMAN: "de", Language.SPANISH: "es", Language.ARABIC: "ar",
    Language.HINDI: "hi", Language.CHINESE: "zh", Language.ITALIAN: "it",
    Language.FRENCH: "fr", Language.HEBREW: "he", Language.JAPANESE: "ja",
}

lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANG_MAP.keys()).build()

@dataclass
class LanguageResult:
    lang: str            
    langdetect: str      
    lingua: str          
    detector: str        

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

    try:
        ld_code = normalize(detect(text))
    except LangDetectException:
        ld_code = "unknown"

    lingua_lang = lingua_detector.detect_language_of(text)
    if lingua_lang is None:
        li_code = "unknown"
    else:
        li_code = LINGUA_LANG_MAP.get(lingua_lang, "unknown")

    if ld_code == li_code and ld_code in ALLOWED_LANGUAGES:
        return LanguageResult(lang=ld_code, langdetect=ld_code, lingua=li_code, detector="agreement")

    inld_lang = run_inld(text)
    if inld_lang:
        return LanguageResult(lang=inld_lang, langdetect=ld_code, lingua=li_code, detector="inld")

    return LanguageResult(lang="other", langdetect=ld_code, lingua=li_code, detector="other")