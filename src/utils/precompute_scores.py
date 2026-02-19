import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.toxicity_judge import ToxicityJudge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "messages_with_language.parquet"
TOX_SCORES_PATH = DATA_DIR / "toxicity_scores.npy"
TOX_META_PATH = DATA_DIR / "toxicity_scores_meta.json"


def load_messages() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run data_ingestion.py first."
        )
    df = pd.read_parquet(PROCESSED_PATH)
    if "message_text" not in df.columns:
        raise ValueError("Column 'message_text' missing from processed dataset.")
    df = df.reset_index(drop=True)
    return df


def compute_scores(texts, batch_size: int = 64) -> np.ndarray:
    print("Loading ToxicityJudge (XLM-R multilingual classifier)...")
    judge = ToxicityJudge()  # auto-selects GPU if available

    scores = []
    n = len(texts)
    print(f"Scoring {n} messages for toxicity...")
    for i in tqdm(range(0, n, batch_size), desc="toxicity scoring"):
        batch = texts[i : i + batch_size]
        batch_scores = judge.score_batch(batch, batch_size=len(batch))
        scores.extend(batch_scores)

    return np.array(scores, dtype="float32")


def main():
    df = load_messages()
    texts = df["message_text"].fillna("").astype(str).tolist()

    scores = compute_scores(texts, batch_size=64)
    print(f"Scores shape: {scores.shape}")

    np.save(TOX_SCORES_PATH, scores)
    print(f"Saved toxicity scores to {TOX_SCORES_PATH}")

    meta = {
        "model_name": "textdetox/xlmr-large-toxicity-classifier-v2",
        "num_messages": int(scores.shape[0]),
        "score_range": [0.0, 1.0],
        "source_file": str(PROCESSED_PATH.name),
    }
    with open(TOX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved toxicity meta to {TOX_META_PATH}")

    print("\n✓ Toxicity score precomputation complete.")


if __name__ == "__main__":
    main()
