"""
Day 29 — Vectorized Simulator Architecture.

Processes states in parallel across N independent simulator instances.
Stack N embeddings → shape (N, 384), query MaskablePPO → predict N actions
simultaneously, apply action vector → advance all N simulators.

RUN:
  python -m src.diagnostics.vectorized_eval
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import time
from collections import defaultdict

from sb3_contrib import MaskablePPO
from src.env.discord_env import DiscordEnv, LANG_TO_IDX, NUM_LANGUAGES


ACTION_NAMES = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]


class VectorizedSimulator:
    """
    Runs N episodes in parallel by batching observations and
    querying the policy once per step across all active episodes.
    """

    def __init__(self, model_path: str = "data/models/best/best_model.zip",
                 data_dir: str = "data/processed", n_parallel: int = 32):
        self.model = MaskablePPO.load(model_path)
        self.data_dir = data_dir
        self.n_parallel = n_parallel

        # Load shared data
        with open(f"{data_dir}/episodes.json", "r") as f:
            self.all_episodes = json.load(f)
        self.embeddings = np.load(f"{data_dir}/context_embeddings.npy")
        self.toxicity_scores = np.load(f"{data_dir}/toxicity_scores.npy")

        print(f"Vectorized simulator: {len(self.all_episodes)} episodes, "
              f"N={n_parallel} parallel")

    def _make_env(self, ep_data):
        """Create a lightweight episode state tracker."""
        return {
            "thread_id": ep_data["thread_id"],
            "step_indices": ep_data["step_indices"],
            "user_ids": ep_data["user_ids"],
            "languages": ep_data.get("languages", ["en"] * len(ep_data["step_indices"])),
            "current_step": 0,
            "ledger": {},
            "recent_tox": [],
            "recent_actions": [],
            "done": False,
            "results": [],
        }

    def _get_obs_and_mask(self, env):
        """Get observation and action mask for a single env."""
        step = env["current_step"]
        idx = env["step_indices"][step]
        tox = float(self.toxicity_scores[idx])
        uid = env["user_ids"][step]
        lang = env["languages"][step] if step < len(env["languages"]) else "en"

        # Ensure ledger
        if uid not in env["ledger"]:
            env["ledger"][uid] = {"warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0}

        led = env["ledger"][uid]

        # Embedding
        emb = np.nan_to_num(self.embeddings[idx].astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        emb = np.clip(emb, -1.0, 1.0)

        # Language one-hot
        lang_vec = np.zeros(NUM_LANGUAGES, dtype=np.float32)
        li = LANG_TO_IDX.get(lang, -1)
        if li >= 0:
            lang_vec[li] = 1.0

        obs = {
            "message_embedding": emb,
            "toxicity_score": np.array([np.clip(tox, 0.0, 1.0)], dtype=np.float32),
            "user_history": np.array([
                min(led["warns"] / 5.0, 1.0),
                min(led["timeouts"] / 3.0, 1.0),
                min(led["total_infractions"] / 10.0, 1.0),
            ], dtype=np.float32),
            "server_heat": np.array([0.0, 0.0], dtype=np.float32),
            "language_id": lang_vec,
        }

        # Action mask
        mask = np.ones(5, dtype=bool)
        if led["timeouts"] < 1:
            mask[4] = False
        if led["total_infractions"] < 2:
            mask[3] = False
        if tox < 0.15:
            mask[2:] = False
        if tox >= 0.85:
            mask[0] = False
        if not mask.any():
            mask[0] = True

        return obs, mask, tox, uid, lang

    def _apply_action(self, env, action, tox, uid):
        """Update ledger and record result."""
        led = env["ledger"][uid]

        if action == 1:
            led["warns"] += 1.0
            led["total_infractions"] += 1.0
        elif action == 2:
            led["total_infractions"] += 1.0
        elif action == 3:
            led["timeouts"] += 1.0
            led["total_infractions"] += 1.0
        elif action == 4:
            led["total_infractions"] += 1.0

        env["results"].append({
            "action": int(action),
            "toxicity": tox,
            "user_id": uid,
        })

        env["current_step"] += 1
        if env["current_step"] >= len(env["step_indices"]):
            env["done"] = True

    def run(self, n_episodes: int = None) -> dict:
        """Run vectorized evaluation across all or n_episodes episodes."""
        if n_episodes is None:
            n_episodes = len(self.all_episodes)
        n_episodes = min(n_episodes, len(self.all_episodes))

        rng = np.random.default_rng(42)
        episode_indices = rng.choice(len(self.all_episodes), size=n_episodes, replace=False)

        # Track metrics
        total_steps = 0
        action_counts = defaultdict(int)
        lang_action_counts = defaultdict(lambda: defaultdict(int))
        false_positives = 0
        false_negatives = 0

        start_time = time.time()

        # Process in batches of n_parallel
        completed = 0
        remaining = list(episode_indices)

        while remaining:
            # Fill batch
            batch_indices = remaining[:self.n_parallel]
            remaining = remaining[self.n_parallel:]
            envs = [self._make_env(self.all_episodes[i]) for i in batch_indices]

            # Step all envs until all done
            while any(not e["done"] for e in envs):
                active = [e for e in envs if not e["done"]]

                for env in active:
                    obs, mask, tox, uid, lang = self._get_obs_and_mask(env)
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                    action = int(action)

                    self._apply_action(env, action, tox, uid)

                    # Track metrics
                    total_steps += 1
                    action_counts[action] += 1
                    lang_action_counts[lang][action] += 1

                    if tox < 0.30 and action > 0:
                        false_positives += 1
                    if tox >= 0.70 and action == 0:
                        false_negatives += 1

            completed += len(batch_indices)

        elapsed = time.time() - start_time
        steps_per_sec = total_steps / max(elapsed, 0.001)

        # Compile results
        results = {
            "episodes": n_episodes,
            "total_steps": total_steps,
            "elapsed_seconds": round(elapsed, 2),
            "steps_per_second": round(steps_per_sec, 1),
            "action_distribution": {ACTION_NAMES[k]: v for k, v in sorted(action_counts.items())},
            "false_positive_rate": round(false_positives / max(total_steps, 1), 4),
            "false_negative_rate": round(false_negatives / max(total_steps, 1), 4),
            "language_action_counts": {
                lang: {ACTION_NAMES[a]: c for a, c in sorted(actions.items())}
                for lang, actions in sorted(lang_action_counts.items())
            },
        }

        # Print report
        print(f"\n{'=' * 70}")
        print(f"VECTORIZED EVALUATION — {n_episodes} episodes, {total_steps} steps")
        print(f"{'=' * 70}")
        print(f"   Throughput: {steps_per_sec:.0f} steps/sec ({elapsed:.1f}s total)")
        print(f"\n   Action Distribution:")
        for a in range(5):
            count = action_counts.get(a, 0)
            pct = count / max(total_steps, 1) * 100
            bar = "█" * int(pct / 2)
            print(f"     {ACTION_NAMES[a]:8s}: {count:5d} ({pct:5.1f}%) {bar}")
        print(f"\n   FP Rate: {results['false_positive_rate']:.4f}")
        print(f"   FN Rate: {results['false_negative_rate']:.4f}")

        return results


if __name__ == "__main__":
    sim = VectorizedSimulator(n_parallel=32)
    results = sim.run()
