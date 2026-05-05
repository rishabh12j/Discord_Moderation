import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "messages_with_language.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMB_META_PATH = DATA_DIR / "embeddings_meta.json"


def load_messages() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run data_ingestion.py (Day 3) first."
        )
    df = pd.read_parquet(PROCESSED_PATH)
    if "message_text" not in df.columns:
        raise ValueError("Column 'message_text' missing from processed dataset.")
    # Ensure stable order
    df = df.reset_index(drop=True)
    return df


def compute_embeddings(texts, batch_size: int = 256) -> np.ndarray:
    print("Loading SentenceTransformer model: paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    model = model.to("cuda")  # will fall back to CPU if no GPU

    print(f"Encoding {len(texts)} messages into embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine distance becomes dot product
    )
    return embeddings.astype("float32")


def main():
    df = load_messages()
    texts = df["message_text"].fillna("").astype(str).tolist()

    embeddings = compute_embeddings(texts, batch_size=256)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    # simple meta file
    meta = {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(embeddings.shape[1]),
        "num_messages": int(embeddings.shape[0]),
        "normalized": True,
        "source_file": str(PROCESSED_PATH.name),
    }

    import json
    with open(EMB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved embedding meta to {EMB_META_PATH}")

    print("\n✓ Embedding precomputation complete.")


if __name__ == "__main__":
    main()
