import numpy as np
import pandas as pd
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "messages_with_language.parquet"
SCORES_PATH = PROJECT_ROOT / "data" / "toxicity_scores.npy"
THRESHOLDS_OUT = PROJECT_ROOT / "data" / "calibrated_thresholds.json"

def main():
    print("Loading data for Judge Tuning...")
    if not PROCESSED_PATH.exists() or not SCORES_PATH.exists():
        raise FileNotFoundError("Processed data or toxicity scores missing. Run Day 3 and Day 5 scripts.")

    df = pd.read_parquet(PROCESSED_PATH)
    scores = np.load(SCORES_PATH)
    
    # Align scores with the dataframe
    df['toxicity_score'] = scores
    
    print(f"\n{'Language':<10} | {'Count':<8} | {'Mean':<6} | {'90th %ile':<9} | {'% > 0.8':<8}")
    print("-" * 55)
    
    calibrated_thresholds = {}
    
    # Group by language to audit the classifier's behavior
    for lang, group in df.groupby('language'):
        count = len(group)
        if count < 50:  # Skip very rare/noise languages
            continue
            
        mean_score = group['toxicity_score'].mean()
        p90 = np.percentile(group['toxicity_score'], 90)
        pct_over_80 = (group['toxicity_score'] > 0.8).mean() * 100
        
        print(f"{lang:<10} | {count:<8} | {mean_score:.3f} | {p90:.3f}     | {pct_over_80:.1f}%")
        
        # Save the 90th percentile as our new calibrated threshold
        # We clip it between 0.7 and 0.95 so we don't accidentally allow extreme toxicity
        calibrated_thresholds[lang] = float(np.clip(p90, 0.70, 0.95))
        
    # Set a fallback for unknown/other languages
    calibrated_thresholds["default"] = 0.80
        
    with open(THRESHOLDS_OUT, "w") as f:
        json.dump(calibrated_thresholds, f, indent=2)
        
    print(f"\nCalibrated thresholds saved to {THRESHOLDS_OUT}")

if __name__ == "__main__":
    main()