import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # from src/utils -> project root
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "messages_with_language.parquet"
USER_NORMS_PATH = DATA_DIR / "user_norms.json"
USER_NORMS_META_PATH = DATA_DIR / "user_norms_meta.json"


def load_messages() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed messages file not found at {PROCESSED_PATH}. "
            f"Run data_ingestion.py (Day 3) first."
        )
    df = pd.read_parquet(PROCESSED_PATH)

    required_cols = {"user_id", "language", "label_toxic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    return df


def compute_user_norms(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute language-agnostic toxicity norms per user
    based on label_toxic (0/1).
    """
    df = df.copy()
    df["label_toxic"] = df["label_toxic"].astype(float)

    grouped = df.groupby("user_id")
    overall_avg = grouped["label_toxic"].mean()
    msg_count = grouped["label_toxic"].size()

    user_norms: Dict[str, Dict[str, Any]] = {}
    for user_id in overall_avg.index:
        user_norms[str(user_id)] = {
            "overall_avg_toxicity": float(overall_avg.loc[user_id]),
            "message_count": int(msg_count.loc[user_id]),
        }

    return user_norms



def compute_global_meta(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Optional: global statistics that will help later to normalize features
    such as message_count_norm or to debug fairness.
    """
    grouped_user = df.groupby("user_id")
    msg_count = grouped_user["label_toxic"].size()
    overall_avg = grouped_user["label_toxic"].mean()

    meta = {
        "num_users": int(msg_count.shape[0]),
        "msg_count_min": int(msg_count.min()),
        "msg_count_max": int(msg_count.max()),
        "msg_count_median": float(msg_count.median()),
        "msg_count_mean": float(msg_count.mean()),
        "overall_toxicity_mean": float(overall_avg.mean()),
    }
    return meta


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    print(f"Loading messages from {PROCESSED_PATH}")
    df = load_messages()
    print(f"Loaded {len(df)} messages for user norm computation")

    print("Computing user norms...")
    user_norms = compute_user_norms(df)
    print(f"Computed norms for {len(user_norms)} users")

    print(f"Saving user norms to {USER_NORMS_PATH}")
    save_json(user_norms, USER_NORMS_PATH)

    print("Computing global meta statistics...")
    meta = compute_global_meta(df)
    print(f"Saving meta stats to {USER_NORMS_META_PATH}")
    save_json(meta, USER_NORMS_META_PATH)

    print("\n✓ User norm computation complete.")


if __name__ == "__main__":
    main()
