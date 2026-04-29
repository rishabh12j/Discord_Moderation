"""
Add language field to context_strings.json.

Strategy:
1. Extract language code from thread_id (e.g., 'toxic-he-001' → 'he')
2. Fall back to Unicode script detection for non-Latin scripts
3. Default remaining Latin-script entries to 'en'

RUN:
  python -m src.pipeline.add_language_field
"""
import os
import json
from collections import Counter


LANG_CODES = {"en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"}

# Aliases found in thread IDs
LANG_ALIASES = {"jp": "ja", "italian": "it"}


def extract_lang_from_thread_id(tid: str) -> str | None:
    """Extract language code from thread_id like 'toxic-he-001', 'ar-benign-1'."""
    parts = tid.split("-")
    for p in parts:
        if p in LANG_CODES:
            return p
        if p in LANG_ALIASES:
            return LANG_ALIASES[p]
    return None


def detect_script_language(text: str) -> str | None:
    """Detect language from Unicode script ranges for non-Latin scripts."""
    if not text.strip():
        return None

    counts = {
        "ar": sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF),
        "hi": sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F),
        "he": sum(1 for c in text if 0x0590 <= ord(c) <= 0x05FF),
        "zh": sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF),
        "ja": sum(1 for c in text if 0x3040 <= ord(c) <= 0x30FF or 0x31F0 <= ord(c) <= 0x31FF),
        "ru": sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF),  # Cyrillic (ru or uk)
    }

    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return None  # Latin script


def add_language_field(
    input_file: str = "data/processed/context_strings.json",
    output_file: str = "data/processed/context_strings.json",  # overwrite in place
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...")

    lang_source = Counter()  # track how we tagged each entry
    lang_dist = Counter()

    for entry in data:
        # Priority 1: thread_id
        lang = extract_lang_from_thread_id(entry["thread_id"])
        if lang:
            lang_source["thread_id"] += 1
        else:
            # Priority 2: script detection
            lang = detect_script_language(entry.get("raw_text", ""))
            if lang:
                lang_source["script"] += 1
            else:
                # Priority 3: default to English (Latin script, no other signal)
                lang = "en"
                lang_source["default_en"] += 1

        entry["language"] = lang
        lang_dist[lang] += 1

    # Report
    print(f"\nLanguage tagging sources:")
    for src, count in sorted(lang_source.items(), key=lambda x: -x[1]):
        print(f"   {src:12s}: {count:5d} ({count/len(data)*100:.1f}%)")

    print(f"\n🌍 Language distribution:")
    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
        print(f"   {lang:4s}: {count:5d} ({count/len(data)*100:.1f}%)")

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(data)} entries with language field to {output_file}")


if __name__ == "__main__":
    add_language_field()
