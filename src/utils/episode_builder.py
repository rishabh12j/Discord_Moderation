from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Any, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_PATH = DATA_DIR / "processed" / "messages_with_language.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TOX_SCORES_PATH = DATA_DIR / "toxicity_scores.npy"


@dataclass
class EpisodeChunk:
    messages: List[str]
    embeddings: np.ndarray
    toxicity_scores: np.ndarray
    languages: List[str]
    user_ids: List[str]
    thread_id: Any


def load_base_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run data_ingestion.py first."
        )
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. Run precompute_embeddings.py first."
        )
    if not TOX_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"{TOX_SCORES_PATH} not found. Run precompute_scores.py first."
        )

    df = pd.read_parquet(PROCESSED_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    scores = np.load(TOX_SCORES_PATH)

    n_df = len(df)
    if embeddings.shape[0] != n_df or scores.shape[0] != n_df:
        raise ValueError(
            f"Length mismatch: df={n_df}, embeddings={embeddings.shape[0]}, "
            f"scores={scores.shape[0]}"
        )

    required_cols = {"message_text", "language", "user_id", "thread_id", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # sort by thread, then timestamp for chronological chunks
    df = df.sort_values(["thread_id", "timestamp"]).reset_index(drop=True)

    return df, embeddings, scores


def episode_generator(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    scores: np.ndarray,
    chunk_size: int = 20,
    min_chunk_size: Optional[int] = None,
) -> Generator[EpisodeChunk, None, None]:
    """
    Yield fixed-length conversation chunks with aligned arrays.
    """
    if min_chunk_size is None:
        min_chunk_size = chunk_size

    # After sorting above, df.index no longer matches original indices.
    # Build a mapping from new index -> embedding/score index.
    # Because all preprocessing scripts reset_index in the same order,
    # the index after sort is just a permutation of 0..N-1.
    index_map = df.index.to_numpy()

    for thread_id, thread in df.groupby("thread_id", sort=False):
        thread = thread.sort_values("timestamp")
        idx = thread.index.to_numpy()

        if len(thread) < min_chunk_size:
            continue

        for start in range(0, len(thread) - chunk_size + 1, chunk_size):
            sel_idx = idx[start : start + chunk_size]

            yield EpisodeChunk(
                messages=thread.loc[sel_idx, "message_text"].tolist(),
                embeddings=embeddings[index_map[sel_idx]],
                toxicity_scores=scores[index_map[sel_idx]],
                languages=thread.loc[sel_idx, "language"].tolist(),
                user_ids=thread.loc[sel_idx, "user_id"].tolist(),
                thread_id=thread_id,
            )


if __name__ == "__main__":
    df, emb, scores = load_base_data()
    print(f"Loaded df={len(df)}, emb={emb.shape}, scores={scores.shape}")

    gen = episode_generator(df, emb, scores, chunk_size=20)
    first_chunk = next(gen, None)

    if first_chunk is None:
        print("No chunks produced. Check thread_id distribution and chunk_size.")
    else:
        print(f"First chunk thread_id={first_chunk.thread_id}")
        print(f"Messages: {len(first_chunk.messages)}")
        print(f"Embeddings shape: {first_chunk.embeddings.shape}")
        print(f"Toxicity shape: {first_chunk.toxicity_scores.shape}")
        print(f"Languages sample: {first_chunk.languages[:5]}")
        print(f"User_ids sample: {first_chunk.user_ids[:5]}")