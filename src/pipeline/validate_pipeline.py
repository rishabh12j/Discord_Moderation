import numpy as np
import json

# Load all outputs
emb = np.load("data/processed/context_embeddings.npy")
tox = np.load("data/processed/toxicity_scores.npy")
with open("data/processed/episodes.json", "r") as f:
    episodes = json.load(f)

total_steps = len(emb)
total_episode_steps = sum(len(e["step_indices"]) for e in episodes)
num_episodes = len(episodes)

print("PIPELINE VALIDATION")
print(f"   Embeddings shape: {emb.shape}")
print(f"   Toxicity shape:   {tox.shape}")
print(f"   Episodes:         {num_episodes}")
print(f"   Total steps:      {total_steps}")
print(f"   Episode steps:    {total_episode_steps}")
print(f"   MATCH: {total_steps == total_episode_steps}")

# Sample first episode
first_episode = episodes[0]
print(f"\nSample Episode:")
print(f"   Thread ID: {first_episode['thread_id']}")
print(f"   Steps: {first_episode['step_indices'][:5]}... (total {len(first_episode['step_indices'])})")
print(f"   Users: {first_episode['user_ids'][:5]}...")
print(f"   Langs: {first_episode['languages'][:5]}...")

print(f"\nToxicity Health Check:")
print(f"   Toxic steps (>0.5): {(tox > 0.5).sum()} / {len(tox)} ({(tox > 0.5).mean()*100:.1f}%)")
print(f"   Toxicity range: [{tox.min():.3f}, {tox.max():.3f}]")
print(f"   Mean toxicity: {tox.mean():.3f}")

