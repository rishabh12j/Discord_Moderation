"""
Fixed training script.
- Multiple parallel envs for sample efficiency
- Reward scaling wrapper (applied LAST, after all penalty wrappers)
- Lagrangian constraint wrapper
- Proper gradient clipping (max_grad_norm=0.5)
- Constraint-aware checkpoint selection (FIX E)
- Train/test episode split (FIX C)
- Larger effective batch size: n_steps=4096, batch_size=256 (FIX B)
"""
import os
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.env.discord_env import DiscordEnv
from src.env.wrappers import RewardScalingWrapper, LagrangianPenaltyWrapper, DisparateImpactWrapper, MetricsTrackingWrapper


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_env(episodes_file: str = "data/processed/episodes_train.json"):
    """Build environment with full wrapper stack.

    Wrapper order matters for reward correctness:
      1. All penalty wrappers (Lagrangian, DisparateImpact) operate in raw reward space
      2. RewardScalingWrapper clips the final sum to [-1, 1] — must be LAST before ActionMasker
         so that penalties are not applied on top of an already-clipped value (which would
         allow total reward to reach -4.0 and break PPO advantage normalization).
    """
    env = DiscordEnv(episodes_file=episodes_file)
    env = LagrangianPenaltyWrapper(env, lambda_init=0.5, fp_threshold=0.05)
    env = DisparateImpactWrapper(env, tau=1.5, window_size=2000)
    env = MetricsTrackingWrapper(env)
    env = RewardScalingWrapper(env, scale_factor=5.0)  # LAST: compress after all penalties
    env = ActionMasker(env, mask_fn)
    return env


class ConstraintAwareEvalCallback(BaseCallback):
    """
    FIX E: Saves best_model based on a constraint-penalized score rather than
    mean reward alone.

    Standard MaskableEvalCallback saves on mean_reward. For a CMDP, a model
    with reward=0.9 but FP_rate=0.08 is strictly worse than reward=0.85 and
    FP_rate=0.02 — the former violates the constraint. This callback uses:

        penalized_score = mean_reward                          if fp_rate <= threshold
        penalized_score = mean_reward - 10 * (fp_rate - threshold)  otherwise

    FP/FN rates come from episode_metrics emitted by MetricsTrackingWrapper.
    """

    def __init__(self, eval_env, eval_freq: int = 5000, n_eval_episodes: int = 20,
                 save_path: str = "data/models/best", fp_threshold: float = 0.05,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.fp_threshold = fp_threshold
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards, fp_rates, fn_rates = [], [], []
        obs = self.eval_env.reset()

        episodes_done = 0
        while episodes_done < self.n_eval_episodes:
            # Get action masks from the vectorised env
            action_masks = self.eval_env.env_method("action_masks")
            masks = np.array(action_masks)
            action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, done, infos = self.eval_env.step(action)

            for info, d in zip(infos, done):
                if d and "episode_metrics" in info:
                    m = info["episode_metrics"]
                    rewards.append(m["cumulative_reward"])
                    fp_rates.append(m["fp_rate"])
                    fn_rates.append(m["fn_rate"])
                    episodes_done += 1

        if not rewards:
            return True

        mean_reward = float(np.mean(rewards))
        mean_fp = float(np.mean(fp_rates))
        mean_fn = float(np.mean(fn_rates))

        # Penalize FP in score — always, not just above threshold.
        # Below threshold: mild penalty so FP=0.01 still scores lower than FP=0.00.
        # Above threshold: steep penalty to strongly prefer constraint-satisfying models.
        if mean_fp <= self.fp_threshold:
            score = mean_reward - 5.0 * mean_fp
        else:
            score = mean_reward - 10.0 * (mean_fp - self.fp_threshold) - 5.0 * self.fp_threshold

        constraint_ok = mean_fp <= self.fp_threshold
        status = "OK" if constraint_ok else "VIOLATED"

        if self.verbose >= 1:
            print(f"\n[ConstraintEval] step={self.n_calls}  "
                  f"reward={mean_reward:.3f}  fp={mean_fp:.4f} [{status}]  "
                  f"fn={mean_fn:.4f}  score={score:.3f}  best={self.best_score:.3f}")

        if score > self.best_score:
            self.best_score = score
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(os.path.join(self.save_path, "best_model"))
            if self.verbose >= 1:
                print(f"   => New best model saved (score={score:.3f})")

        return True


class ModerationLogCallback(BaseCallback):
    """Log moderation-specific metrics every N steps."""
    
    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_fp_rates = []
        self.episode_fn_rates = []

    def _on_step(self) -> bool:
        # Collect episode metrics from info dicts
        for info in self.locals.get("infos", []):
            if "episode_metrics" in info:
                metrics = info["episode_metrics"]
                self.episode_rewards.append(metrics["cumulative_reward"])
                self.episode_fp_rates.append(metrics["fp_rate"])
                self.episode_fn_rates.append(metrics["fn_rate"])

        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            print(f"\nStep {self.n_calls}:")
            print(f"   Avg Reward:  {np.mean(self.episode_rewards[-100:]):.3f}")
            print(f"   Avg FP Rate: {np.mean(self.episode_fp_rates[-100:]):.3f}")
            print(f"   Avg FN Rate: {np.mean(self.episode_fn_rates[-100:]):.3f}")

        return True


def train():
    print("=" * 60)
    print("DISCORD MODERATION RL AGENT — TWO-STAGE TRAINING")
    print("=" * 60)

    N_ENVS = 4
    TOTAL_TIMESTEPS = 1_000_000
    BC_INIT_PATH = "data/models/bc_init.zip"

    # Build environments — training on train split, eval on held-out test split
    env = make_vec_env(lambda: make_env("data/processed/episodes_train.json"), n_envs=N_ENVS)
    eval_env = make_vec_env(lambda: make_env("data/processed/episodes_test.json"), n_envs=1)

    # Stage 2: RL fine-tuning.
    # If a behavior-cloned init exists, load it and fine-tune (two-stage: BC → RL).
    # Otherwise fall back to RL-Only from random init (single-stage).
    if os.path.exists(BC_INIT_PATH):
        print(f"\n[Stage 2] Loading BC pre-trained weights from {BC_INIT_PATH}")
        model = MaskablePPO.load(
            BC_INIT_PATH,
            env=env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.95,
            max_grad_norm=0.5,
            ent_coef=0.01,
            verbose=1,
            device="auto",
        )
        print("  BC init loaded — RL will refine beyond the oracle rule.")
    else:
        print(f"\n[Single-stage] No BC init found at {BC_INIT_PATH}.")
        print("  Run `python -m src.agent.behavior_clone` first for two-stage training.")
        print("  Falling back to RL-Only from random initialization.\n")
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.95,
            max_grad_norm=0.5,
            ent_coef=0.01,
            verbose=1,
            device="auto",
        )

    # Callbacks
    os.makedirs("data/models/best", exist_ok=True)

    # FIX E: constraint-aware checkpoint — saves best model on reward AND FP constraint,
    # not reward alone. Eval runs on the held-out test split (episodes_test.json).
    eval_callback = ConstraintAwareEvalCallback(
        eval_env=eval_env,
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=80,
        save_path="data/models/best",
        fp_threshold=0.05,
        verbose=1,
    )

    log_callback = ModerationLogCallback(log_freq=5000)

    # Train
    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps with {N_ENVS} parallel envs...")
    print(f"   gamma={model.gamma}, lr={model.learning_rate}, max_grad_norm={model.max_grad_norm}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, log_callback],
        progress_bar=True,
    )

    # Save
    save_path = "data/models/ppo_moderation_fixed"
    model.save(save_path)
    print(f"\nTRAINING COMPLETE! Model saved to {save_path}")


if __name__ == "__main__":
    train()
