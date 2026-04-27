import os
import pandas as pd
from datasets import load_dataset
from sklearn.utils import resample
from collections import Counter

def ingest_multilingual_balanced(
    output_file: str = "data/raw/balanced_multilingual_jigsaw.csv",
    min_samples_per_lang: int = 5000  # Minimum samples per language
):
    """
    Ingests Kaggle + Hugging Face datasets with language-stratified balancing.
    Ensures no single language dominates the final distribution.
    """
    dfs = []
    
    # 1. Load Kaggle datasets (primarily English)
    kaggle_paths = [
        "data/raw/jigsaw-toxic-comment-train.csv",
        "data/raw/jigsaw-unintended-bias-train.csv"
    ]
    
    for path in kaggle_paths:
        if os.path.exists(path):
            print(f"Loading {path}...")
            df = pd.read_csv(path, usecols=["comment_text", "toxic"])
            df.rename(columns={"comment_text": "text"}, inplace=True)
            df["language"] = "en"  # Tag as English
            dfs.append(df)
    
    # 2. Load Hugging Face multilingual dataset (ALL language splits)
    print("Loading textdetox/multilingual_toxicity_dataset from Hugging Face...")
    hf_splits = ["en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"]
    
    for lang in hf_splits:
        try:
            hf_ds = load_dataset("textdetox/multilingual_toxicity_dataset", split=lang)
            df_hf = hf_ds.to_pandas()
            
            # Dynamically detect text and toxicity columns
            text_col = "text" if "text" in df_hf.columns else df_hf.columns[0]
            tox_col = "toxicity" if "toxicity" in df_hf.columns else df_hf.columns[1]
            
            df_hf_clean = df_hf[[text_col, tox_col]].copy()
            df_hf_clean.rename(columns={text_col: "text"}, inplace=True)
            
            # Binarize toxicity
            if df_hf_clean[tox_col].dtype in ["float64", "float32"]:
                df_hf_clean["toxic"] = (df_hf_clean[tox_col] >= 0.5).astype(int)
            else:
                df_hf_clean["toxic"] = df_hf_clean[tox_col].astype(int)
            
            if tox_col != "toxic":
                df_hf_clean.drop(columns=[tox_col], inplace=True)
            
            df_hf_clean["language"] = lang
            dfs.append(df_hf_clean)
            print(f"Successfully loaded {len(df_hf_clean)} rows from HF split '{lang}'.")
        
        except Exception as e:
            print(f"Failed to load HF split '{lang}': {e}")
    
    if not dfs:
        raise ValueError("No data sources were successfully loaded.")
    
    # 3. Merge and deduplicate
    print("Merging datasets and deduplicating...")
    combined = pd.concat(dfs, ignore_index=True)
    combined.dropna(subset=["text", "toxic"], inplace=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 10]
    combined.drop_duplicates(subset=["text"], inplace=True)
    
    print(f"Total after deduplication: {len(combined)} rows")
    
    # 4. Language-stratified balancing
    print("Applying language-stratified balancing...")
    balanced_dfs = []
    
    lang_counts = combined["language"].value_counts()
    print(f"\nLanguage distribution before balancing:")
    print(lang_counts)
    
    for lang in combined["language"].unique():
        lang_df = combined[combined["language"] == lang]
        toxic_lang = lang_df[lang_df["toxic"] == 1]
        benign_lang = lang_df[lang_df["toxic"] == 0]
        
        # Balance within each language
        min_class = min(len(toxic_lang), len(benign_lang))
        
        # Skip languages with insufficient data
        if min_class < 100:
            print(f"Skipping {lang}: insufficient data ({min_class} samples)")
            continue
        
        # Cap at min_samples_per_lang to prevent English dominance
        sample_size = min(min_class, min_samples_per_lang)
        
        toxic_sampled = resample(toxic_lang, replace=False, n_samples=sample_size, random_state=42)
        benign_sampled = resample(benign_lang, replace=False, n_samples=sample_size, random_state=42)
        
        balanced_lang = pd.concat([toxic_sampled, benign_sampled], ignore_index=True)
        balanced_dfs.append(balanced_lang)
        print(f"{lang}: {sample_size * 2} balanced rows (toxic={sample_size}, benign={sample_size})")
    
    # 5. Final concatenation and shuffle
    final_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal language distribution:")
    print(final_df["language"].value_counts())
    print(f"\nFinal toxicity distribution:")
    print(final_df["toxic"].value_counts())
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"\nExtraction complete. Saved {len(final_df)} balanced multilingual rows to {output_file}.")

if __name__ == "__main__":
    ingest_multilingual_balanced(min_samples_per_lang=5000)
