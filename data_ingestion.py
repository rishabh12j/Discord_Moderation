import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.utils.language_utils import detect_language_code, SUPPORTED_LANGUAGES

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_textdetox_dataset() -> pd.DataFrame:
    """
    Load the multilingual toxicity dataset.
    Columns (HF mirror): text, toxic (0/1), language, labels.[web:68]
    """
    print("Loading TextDetox multilingual toxicity dataset...")
    ds = load_dataset("gravitee-io/textdetox-multilingual-toxicity-dataset")
    # single split called 'train' in the mirror
    df = ds["train"].to_pandas()

    # Standardize column names
    df = df.rename(columns={
        "text": "message_text",
        "toxic": "label_toxic",
        "language": "language_original"
    })

    # Keep only useful columns
    df = df[["message_text", "label_toxic", "language_original"]]
    return df


def add_detected_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply your detector and add 'language' and 'language_supported' columns.
    """
    print("Detecting languages (this may take a few minutes)...")
    tqdm.pandas(desc="language detection")
    df["language"] = df["message_text"].progress_apply(detect_language_code)
    df["language_supported"] = df["language"].isin(SUPPORTED_LANGUAGES).astype(int)
    return df


def add_synthetic_conversation_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic user_id, thread_id, and timestamp so the RL environment
    can treat this as threaded conversations later.
    """
    # Simple, reproducible pseudo-conversations:
    # - group consecutive messages into threads of size 20
    # - assign user_id cycling over a fixed pool
    print("Adding synthetic conversation structure...")

    df = df.reset_index(drop=True)
    n = len(df)

    # thread of 20 messages
    df["thread_id"] = (df.index // 20).astype("int64")

    # simple deterministic user ids
    df["user_id"] = (df.index % 1000).map(lambda i: f"user_{i:04d}")

    # fake timestamps: one-second increments from an arbitrary origin
    base_ts = pd.Timestamp("2024-01-01 00:00:00")
    df["timestamp"] = df.index.map(lambda i: base_ts + pd.Timedelta(seconds=int(i)))

    return df


def log_language_distribution(df: pd.DataFrame):
    print("\n=== Language distribution (detected) ===")
    counts = df["language"].value_counts()
    print(counts)

    print("\n=== Language distribution (percentage) ===")
    pct = (counts / counts.sum()).round(4)
    print(pct)


def main():
    df = load_textdetox_dataset()

    print(f"Loaded {len(df)} rows")

    # Add detected language
    df = add_detected_language(df)

    # Add synthetic thread/user/timestamp
    df = add_synthetic_conversation_fields(df)

    # Log distribution
    log_language_distribution(df)

    # Save raw+processed copies
    raw_path = RAW_DIR / "textdetox_multilingual_raw.parquet"
    proc_path = PROCESSED_DIR / "messages_with_language.parquet"

    print(f"\nSaving raw dataset to: {raw_path}")
    df.to_parquet(raw_path, index=False)

    print(f"Saving processed dataset to: {proc_path}")
    df.to_parquet(proc_path, index=False)

    print("\n✓ Data ingestion complete.")


if __name__ == "__main__":
    main()
