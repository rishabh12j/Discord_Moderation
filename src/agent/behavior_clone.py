"""
Behavior Cloning Pre-Training (Stage 1 of two-stage training).

Motivation (Meta paper, Section 3.2.3):
  RL-Only training without a behavioral prior is unstable — the policy
  fails to converge to a coherent reasoning schema. SFT→RL consistently
  outperforms RL-Only: the supervised stage anchors the model in the
  correct behavioral structure, and RL then refines from that starting point.

What this script does:
  1. Rolls out all training episodes using the ORACLE (_get_best_action rule)
  2. Collects (observation, best_action) pairs — this is the supervised dataset
  3. Trains the MaskablePPO policy network via cross-entropy loss (NLL minimization)
     using policy.evaluate_actions(), which is the SB3-native way to get log-probs
  4. Saves the pre-trained weights as data/models/bc_init.zip

The RL training script (train.py) then loads bc_init.zip as its starting policy
instead of random initialization.

RUN:
  python -m src.agent.behavior_clone
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from torch.optim import Adam

sys.path.append(str(Path(__file__).parent.parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

from src.env.discord_env import DiscordEnv
from src.env.wrappers import RewardScalingWrapper, LagrangianPenaltyWrapper, DisparateImpactWrapper, MetricsTrackingWrapper


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def collect_bc_dataset(episodes_file: str = "data/processed/episodes_train.json"):
    """
    Roll out every training episode using the oracle policy (_get_best_action).
    Returns lists of observations, actions, and action masks.
    """
    env = DiscordEnv(episodes_file=episodes_file)
    obs_list, action_list, mask_list = [], [], []

    print(f"Collecting BC dataset from {len(env.episodes)} episodes...")
    for _ in range(len(env.episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            # Compute oracle action BEFORE stepping (ledger state must match)
            idx = env.current_episode.step_indices[env.current_step_in_episode]
            tox = float(env.toxicity_scores[idx])
            uid = env.current_episode.user_ids[env.current_step_in_episode]
            env._ensure_ledger(uid)
            best_action = env._get_best_action(tox, uid)
            mask = env.action_masks()

            obs_list.append({k: v.copy() for k, v in obs.items()})
            action_list.append(best_action)
            mask_list.append(mask.copy())

            obs, _, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated

    env.close()
    print(f"  Collected {len(obs_list)} (obs, action) pairs")

    # Action distribution in the BC dataset
    from collections import Counter
    dist = Counter(action_list)
    names = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]
    print("  Action distribution:")
    for a, name in enumerate(names):
        pct = dist[a] / len(action_list) * 100
        print(f"    {name:8s}: {dist[a]:5d} ({pct:.1f}%)")

    return obs_list, action_list, mask_list


def train_bc(
    obs_list: list,
    action_list: list,
    mask_list: list,
    output_path: str = "data/models/bc_init.zip",
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
):
    """
    Train the MaskablePPO policy via behavior cloning.

    Uses policy.evaluate_actions() which returns log-probabilities of
    taking the given actions under the current policy. Minimizing the
    negative log-likelihood is equivalent to cross-entropy / imitation learning.
    """
    # Build a wrapped env to create the policy with the correct architecture
    def _make_env():
        e = DiscordEnv(episodes_file="data/processed/episodes_train.json")
        e = RewardScalingWrapper(e)
        e = ActionMasker(e, mask_fn)
        return e

    model = MaskablePPO("MultiInputPolicy", _make_env(), verbose=0, device="auto")
    policy = model.policy
    optimizer = Adam(policy.parameters(), lr=lr)

    n_samples = len(obs_list)
    actions_np = np.array(action_list, dtype=np.int64)
    masks_np = np.array(mask_list, dtype=bool)

    print(f"\nBehavior cloning: {n_samples} samples, {n_epochs} epochs, batch={batch_size}")
    print(f"  Device: {next(policy.parameters()).device}")

    best_acc = 0.0
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue

            # Build observation tensors — each key stacked into a batch
            batch_obs = {
                k: torch.tensor(
                    np.stack([obs_list[i][k] for i in idx]),
                    dtype=torch.float32,
                    device=policy.device,
                )
                for k in obs_list[0].keys()
            }
            batch_actions = torch.tensor(actions_np[idx], dtype=torch.long, device=policy.device)
            batch_masks = torch.tensor(masks_np[idx], dtype=torch.bool, device=policy.device)

            # evaluate_actions returns (values, log_prob, entropy)
            # log_prob is the log-probability of taking batch_actions under the current policy
            _, log_prob, _ = policy.evaluate_actions(
                batch_obs, batch_actions, action_masks=batch_masks
            )

            # NLL loss = behavior cloning objective
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Epoch accuracy: fraction of steps where argmax matches oracle
        with torch.no_grad():
            all_obs = {
                k: torch.tensor(
                    np.stack([o[k] for o in obs_list]),
                    dtype=torch.float32,
                    device=policy.device,
                )
                for k in obs_list[0].keys()
            }
            all_masks = torch.tensor(masks_np, dtype=torch.bool, device=policy.device)
            # Get action distributions
            dist = policy.get_distribution(all_obs, action_masks=all_masks)
            preds = dist.distribution.probs.argmax(dim=1).cpu().numpy()
            acc = (preds == actions_np).mean()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={acc:.3f}", end="")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.save(output_path)
            print(f"  ← saved (best acc)")
        else:
            print()

    print(f"\nBehavior cloning complete. Best accuracy: {best_acc:.3f}")
    print(f"Pre-trained model saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("STAGE 1: BEHAVIOR CLONING PRE-TRAINING")
    print("=" * 60)

    obs_list, action_list, mask_list = collect_bc_dataset()
    train_bc(obs_list, action_list, mask_list)

    print("\nNext step: run RL training with the pre-trained weights:")
    print("  python -m src.agent.train")


if __name__ == "__main__":
    main()
