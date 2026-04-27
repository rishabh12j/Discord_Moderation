"""
Per-Language Toxicity Calibration (Change 1).

Problem: The XLM-R classifier produces different confidence distributions 
per language. Arabic toxic content might score 0.55 while English scores 0.95 
for semantically equivalent text. All downstream thresholds (0.30, 0.85) are
calibrated against English behavior.

Solution: Compute per-language percentile anchors from the training data.
For each language, find the classifier's score at the 25th percentile of 
benign content (low anchor) and 75th percentile of toxic content (high anchor).
Then apply a per-language affine transform so that these anchors map to
consistent calibrated values across all languages.

Calibration formula per language:
    calibrated = (raw - low_anchor) / (high_anchor - low_anchor)
    calibrated = clip(calibrated, 0.0, 1.0)

The calibration params are saved to toxicity_calibration.json so the 
production moderator can apply the identical transform at inference time.

RUN:
  python -m src.pipeline.precompute_toxicity
"""
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline


LANGUAGES = ["en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"]


def compute_toxicity_scores(
    input_file: str = "data/processed/context_strings.json",
    output_file: str = "data/processed/toxicity_scores.npy",
    calibration_file: str = "data/processed/toxicity_calibration.json",
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}")

    print("Loading context strings...")
    with open(input_file, "r", encoding="utf-8") as f:
        states = json.load(f)

    raw_texts = [state.get("raw_text", "") for state in states]
    languages = [state.get("language", "other") for state in states]
    # Ground truth toxic labels (if available from data_ingestion)
    is_toxic_labels = [state.get("is_toxic", None) for state in states]

    print(f"Loading textdetox/xlmr-large-toxicity-classifier-v2...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier-v2",
        device=device,
        truncation=True,
        max_length=512,
    )

    # ── Step 1: Score all messages (raw) ────────────────────────
    print(f"Scoring {len(raw_texts)} messages...")
    raw_scores = []
    batch_size = 32

    for i in tqdm(range(0, len(raw_texts), batch_size)):
        batch = raw_texts[i : i + batch_size]
        batch = [t if t.strip() else "neutral" for t in batch]
        results = classifier(batch)

        for j, res in enumerate(results):
            is_toxic_label = res["label"].lower() in ["toxic", "label_1", "1"]
            score = res["score"] if is_toxic_label else 1.0 - res["score"]
            if not raw_texts[i + j].strip():
                score = 0.0
            raw_scores.append(float(np.clip(score, 0.0, 1.0)))

    raw_scores = np.array(raw_scores, dtype=np.float32)
    languages = np.array(languages)

    # ── Step 2: Compute per-language calibration anchors ────────
    print(f"\n📊 Computing per-language calibration anchors...")

    calibration_params = {"method": "percentile_anchors", "languages": {}}

    # Use English as the reference language
    en_mask = languages == "en"
    en_scores = raw_scores[en_mask]

    if len(en_scores) > 0:
        # English anchors define the target scale
        ref_low = float(np.percentile(en_scores[en_scores < 0.5], 75)) if (en_scores < 0.5).sum() > 10 else 0.10
        ref_high = float(np.percentile(en_scores[en_scores >= 0.5], 75)) if (en_scores >= 0.5).sum() > 10 else 0.90
    else:
        ref_low = 0.10
        ref_high = 0.90

    calibration_params["reference_language"] = "en"
    calibration_params["reference_low"] = ref_low
    calibration_params["reference_high"] = ref_high

    print(f"   Reference (English): low={ref_low:.3f}, high={ref_high:.3f}")

    calibrated_scores = raw_scores.copy()

    for lang in LANGUAGES:
        lang_mask = languages == lang
        lang_scores = raw_scores[lang_mask]

        if len(lang_scores) < 20:
            print(f"   {lang:4s}: insufficient data ({len(lang_scores)} samples) — no calibration")
            calibration_params["languages"][lang] = {
                "count": int(len(lang_scores)),
                "raw_low": None,
                "raw_high": None,
                "calibrated": False,
            }
            continue

        # Compute this language's anchors
        benign_scores = lang_scores[lang_scores < 0.5]
        toxic_scores = lang_scores[lang_scores >= 0.5]

        if len(benign_scores) < 10 or len(toxic_scores) < 10:
            print(f"   {lang:4s}: skewed distribution (benign={len(benign_scores)}, "
                  f"toxic={len(toxic_scores)}) — using global anchors")
            lang_low = ref_low
            lang_high = ref_high
        else:
            lang_low = float(np.percentile(benign_scores, 75))
            lang_high = float(np.percentile(toxic_scores, 75))

        # Avoid degenerate case
        if lang_high - lang_low < 0.05:
            lang_low = ref_low
            lang_high = ref_high

        # Apply affine transform: map [lang_low, lang_high] → [ref_low, ref_high]
        scale = (ref_high - ref_low) / max(lang_high - lang_low, 0.01)
        offset = ref_low - lang_low * scale

        calibrated = raw_scores[lang_mask] * scale + offset
        calibrated = np.clip(calibrated, 0.0, 1.0)
        calibrated_scores[lang_mask] = calibrated

        # Stats
        raw_mean = float(lang_scores.mean())
        cal_mean = float(calibrated.mean())
        shift = cal_mean - raw_mean

        calibration_params["languages"][lang] = {
            "count": int(len(lang_scores)),
            "raw_low": float(lang_low),
            "raw_high": float(lang_high),
            "scale": float(scale),
            "offset": float(offset),
            "raw_mean": float(raw_mean),
            "calibrated_mean": float(cal_mean),
            "shift": float(shift),
            "calibrated": True,
        }

        flag = "⚠️" if abs(shift) > 0.05 else "  "
        print(f"   {lang:4s}: n={len(lang_scores):5d}  "
              f"raw_low={lang_low:.3f}  raw_high={lang_high:.3f}  "
              f"scale={scale:.3f}  shift={shift:+.3f} {flag}")

    # Handle "other" language (no calibration)
    other_mask = ~np.isin(languages, LANGUAGES)
    if other_mask.sum() > 0:
        calibration_params["languages"]["other"] = {
            "count": int(other_mask.sum()),
            "calibrated": False,
        }

    # ── Step 3: Report ──────────────────────────────────────────
    print(f"\n📊 Calibrated Score Distribution:")
    print(f"   Range:        [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]")
    print(f"   Mean:         {calibrated_scores.mean():.3f}")
    print(f"   Toxic (>0.5): {(calibrated_scores > 0.5).sum()} / {len(calibrated_scores)} "
          f"({(calibrated_scores > 0.5).mean()*100:.1f}%)")

    # Per-language distribution after calibration
    print(f"\n📊 Per-Language Calibrated Distribution:")
    print(f"   {'Lang':5s} {'Count':>6s} {'Mean':>7s} {'<0.30':>7s} {'0.30-0.70':>10s} {'>0.70':>7s}")
    print(f"   {'─'*48}")
    for lang in LANGUAGES:
        lang_mask = languages == lang
        ls = calibrated_scores[lang_mask]
        if len(ls) == 0:
            continue
        low_pct = (ls < 0.30).mean() * 100
        mid_pct = ((ls >= 0.30) & (ls < 0.70)).mean() * 100
        high_pct = (ls >= 0.70).mean() * 100
        print(f"   {lang:5s} {len(ls):6d} {ls.mean():7.3f} {low_pct:6.1f}% {mid_pct:9.1f}% {high_pct:6.1f}%")

    # ── Step 4: Save ────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, calibrated_scores)

    os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
    with open(calibration_file, "w") as f:
        json.dump(calibration_params, f, indent=2)

    print(f"\n✅ Saved calibrated scores to {output_file}")
    print(f"✅ Saved per-language calibration params to {calibration_file}")


if __name__ == "__main__":
    compute_toxicity_scores()
