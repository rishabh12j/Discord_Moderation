import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json


class NumpyEncoder(json.JSONEncoder):
    """Converts numpy types to native Python for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def find_optimal_k(embeddings, lang, max_k: int = 20) -> int:
    """Find optimal k using elbow + silhouette analysis."""
    inertias = []
    silhouettes = []

    # Upper bound on k so we do not over-fragment small langs
    max_k_effective = min(max_k, max(2, len(embeddings) // 10))

    for k in range(2, max_k_effective + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(embeddings, labels))

    # If we somehow did not compute anything, fall back
    if len(inertias) < 3:
        print(f"{lang}: fallback k=2 (insufficient points for elbow)")
        return 2

    # Elbow method: biggest drop in inertia second derivative
    elbow_k = int(np.argmin(np.diff(np.diff(inertias))) + 2)

    # Silhouette peak
    silhouette_peak_k = int(np.argmax(silhouettes) + 2)

    optimal_k = min(elbow_k, silhouette_peak_k, max_k_effective)
    print(f"{lang}: Optimal k={optimal_k} (elbow={elbow_k}, silhouette_peak={silhouette_peak_k})")

    return optimal_k


def cluster_multilingual(
    input_file: str = "data/raw/balanced_multilingual_jigsaw.csv",
    output_file: str = "data/processed/clustered_multilingual.csv",
    metrics_file: str = "data/logs/clustering_metrics.json",
    max_clusters_per_language: int = 15,
    auto_tune_k: bool = True,
):
    """
    K-Means clustering per language with automatic k selection and quality metrics.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Loading sentence transformer 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    clustered_dfs = []
    clustering_metrics = {}

    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang].copy()
        print(f"\n{'=' * 60}")
        print(f"Processing {lang}: {len(lang_df)} samples")

        texts = lang_df["text"].astype(str).tolist()
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Choose k
        if auto_tune_k:
            optimal_k = find_optimal_k(embeddings, lang, max_clusters_per_language)
        else:
            optimal_k = min(max_clusters_per_language, max(2, len(lang_df) // 20))

        if optimal_k < 2:
            print(f"Skipping clustering for {lang}: insufficient samples")
            lang_df["cluster_id"] = 0
            clustering_metrics[lang] = {
                "n_samples": int(len(lang_df)),
                "n_clusters": 1,
                "silhouette_score": None,
                "davies_bouldin_score": None,
                "toxic_ratio": float(lang_df["toxic"].mean()) if "toxic" in lang_df.columns else 0.0,
                "quality": "skipped",
                "status": "skipped_insufficient_data",
            }
            clustered_dfs.append(lang_df)
            continue

        print(f"Using optimal k={optimal_k}")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        lang_df["cluster_id"] = cluster_labels

        # Metrics
        try:
            silhouette_avg = float(silhouette_score(embeddings, cluster_labels))
            davies_bouldin = float(davies_bouldin_score(embeddings, cluster_labels))
            toxic_ratio = float(lang_df["toxic"].mean()) if "toxic" in lang_df.columns else 0.0

            if toxic_ratio > 0.6:
                print(f"{lang} toxicity-heavy ({toxic_ratio:.1%}) → low silhouette is expected")
                quality = "Normal for toxic data" if silhouette_avg > -0.1 else "Review data"
            elif silhouette_avg > 0.3:
                quality = "Excellent"
            elif silhouette_avg > 0.1:
                quality = "Good"
            else:
                quality = "Acceptable for text"

            print(f"Metrics: Silhouette={silhouette_avg:.3f}, DBI={davies_bouldin:.3f}")
            print(f"Quality: {quality}")

            clustering_metrics[lang] = {
                "n_samples": int(len(lang_df)),
                "n_clusters": int(optimal_k),
                "silhouette_score": silhouette_avg,
                "davies_bouldin_score": davies_bouldin,
                "toxic_ratio": toxic_ratio,
                "quality": quality,
                "status": "success",
            }
        except Exception as e:
            print(f"Metrics error for {lang}: {e}")
            clustering_metrics[lang] = {
                "n_samples": int(len(lang_df)),
                "n_clusters": int(optimal_k),
                "silhouette_score": None,
                "davies_bouldin_score": None,
                "toxic_ratio": float(lang_df["toxic"].mean()) if "toxic" in lang_df.columns else 0.0,
                "quality": "metrics_failed",
                "status": "metrics_failed",
                "error": str(e),
            }

        # Prefix cluster IDs
        lang_df["cluster_id"] = [f"{lang}_{i}" for i in cluster_labels]
        clustered_dfs.append(lang_df)

    # Merge all languages
    final_df = pd.concat(clustered_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)

    # Build JSON-safe metrics summary
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics_summary = {
        "summary": {
            "total_clusters": int(final_df["cluster_id"].nunique()),
            "avg_clusters_per_lang": float(
                np.mean([m.get("n_clusters", 0) for m in clustering_metrics.values()])
            )
            if clustering_metrics
            else 0.0,
            "avg_silhouette": float(
                np.mean(
                    [
                        m.get("silhouette_score")
                        for m in clustering_metrics.values()
                        if m.get("silhouette_score") is not None
                    ]
                )
            )
            if any(m.get("silhouette_score") is not None for m in clustering_metrics.values())
            else 0.0,
        },
        "per_language": {
            lang: {
                "n_samples": int(data.get("n_samples", 0)),
                "n_clusters": int(data.get("n_clusters", 0)),
                "silhouette_score": float(data["silhouette_score"])
                if data.get("silhouette_score") is not None
                else None,
                "davies_bouldin_score": float(data["davies_bouldin_score"])
                if data.get("davies_bouldin_score") is not None
                else None,
                "toxic_ratio": float(data.get("toxic_ratio", 0.0)),
                "quality": data.get("quality", "unknown"),
                "status": data.get("status", "unknown"),
            }
            for lang, data in clustering_metrics.items()
        },
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, cls=NumpyEncoder)

    print("\nFIXED CLUSTERING COMPLETE!")
    print(f"   Data: {len(final_df)} rows, {final_df['cluster_id'].nunique()} clusters")
    print(f"   Metrics: {metrics_file}")


if __name__ == "__main__":
    cluster_multilingual(auto_tune_k=True, max_clusters_per_language=15)
