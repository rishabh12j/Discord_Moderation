import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def compute_context_embeddings(input_file: str = "data/processed/context_strings.json",
                               output_file: str = "data/processed/context_embeddings.npy",
                               batch_size: int = 256):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}.")

    with open(input_file, 'r', encoding='utf-8') as f:
        states = json.load(f)

    context_strings = [state.get("context_string", "") for state in states]

    print("Loading 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"Generating embeddings for {len(context_strings)} context windows...")
    embeddings = model.encode(
        context_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    assert embeddings.shape[1] == 384, f"Unexpected dimension: {embeddings.shape[1]}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, embeddings)

    print(f"Precomputation complete. Saved {embeddings.shape} to {output_file}.")

if __name__ == "__main__":
    compute_context_embeddings()