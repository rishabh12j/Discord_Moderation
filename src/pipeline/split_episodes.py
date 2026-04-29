"""
Train/Test Episode Split.

Splits episodes.json into:
  - episodes_train.json  (80%)
  - episodes_test.json   (20%)

Stratified by thread type so each split has proportional coverage of
toxic / escalating / mild / subtle / benign threads.

RUN (once, before training):
  python -m src.pipeline.split_episodes
"""
import json
import os
import random
from collections import defaultdict

DATA_DIR = "data/processed"


def split_episodes(
    input_file: str = f"{DATA_DIR}/episodes.json",
    train_file: str = f"{DATA_DIR}/episodes_train.json",
    test_file:  str = f"{DATA_DIR}/episodes_test.json",
    test_ratio: float = 0.20,
    seed: int = 42,
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file} — run episode_mapping first.")

    with open(input_file, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    print(f"📂 Loaded {len(episodes)} episodes from {input_file}")

    # Group by thread_type for stratified split
    by_type = defaultdict(list)
    for ep in episodes:
        thread_type = ep.get("thread_type", "unknown")
        by_type[thread_type].append(ep)

    rng = random.Random(seed)
    train_eps, test_eps = [], []

    print("\nStratified split:")
    print(f"   {'Type':15s} {'Total':>6s} {'Train':>6s} {'Test':>6s}")
    print(f"   {'─'*36}")

    for thread_type, group in sorted(by_type.items()):
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        test_eps.extend(group[:n_test])
        train_eps.extend(group[n_test:])
        print(f"   {thread_type:15s} {len(group):6d} {len(group)-n_test:6d} {n_test:6d}")

    # Handle episodes with no thread_type by putting them in train
    print(f"   {'─'*36}")
    print(f"   {'TOTAL':15s} {len(episodes):6d} {len(train_eps):6d} {len(test_eps):6d}")
    print(f"\n   Train: {len(train_eps)/len(episodes)*100:.1f}%  "
          f"Test: {len(test_eps)/len(episodes)*100:.1f}%")

    # Shuffle both splits
    rng.shuffle(train_eps)
    rng.shuffle(test_eps)

    # Save
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_eps, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(train_eps)} train episodes → {train_file}")

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_eps, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(test_eps)} test  episodes → {test_file}")


if __name__ == "__main__":
    split_episodes()
