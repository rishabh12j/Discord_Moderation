"""
Evaluation script for the trained moderation agent.
Tests escalation behavior, fairness, and edge cases.
"""
import numpy as np
import json
from sb3_contrib import MaskablePPO
from src.env.discord_env import DiscordEnv
from src.env.wrappers import MetricsTrackingWrapper
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym


ACTION_NAMES = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def evaluate_on_episodes(model_path: str, n_episodes: int = 100):
    """Run deterministic evaluation on real data episodes."""
    env = DiscordEnv()
    env = MetricsTrackingWrapper(env)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path)

    results = []
    action_totals = {i: 0 for i in range(5)}
    total_fp = 0
    total_fn = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if "episode_metrics" in info:
            m = info["episode_metrics"]
            results.append(m)
            for a, count in m["action_distribution"].items():
                action_totals[int(a)] += count
            total_fp += m["false_positives"]
            total_fn += m["false_negatives"]
            total_steps += m["length"]

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({n_episodes} episodes, {total_steps} steps)")
    print(f"{'='*60}")
    
    print(f"\nAction Distribution:")
    for a in range(5):
        pct = action_totals[a] / max(total_steps, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"   {ACTION_NAMES[a]:8s}: {action_totals[a]:5d} ({pct:5.1f}%) {bar}")
    
    avg_reward = np.mean([r["cumulative_reward"] for r in results])
    print(f"\nAvg Episode Reward: {avg_reward:.3f}")
    print(f"   FP Rate: {total_fp / max(total_steps, 1):.4f}")
    print(f"   FN Rate: {total_fn / max(total_steps, 1):.4f}")

    # Fairness check
    print(f"\n🌍 Language Fairness Check:")
    lang_bans = {}
    lang_totals = {}
    for r in results:
        for lang, actions in r.get("language_actions", {}).items():
            lang_totals[lang] = lang_totals.get(lang, 0) + sum(actions.values())
            lang_bans[lang] = lang_bans.get(lang, 0) + actions.get(4, 0)

    if lang_totals:
        overall_ban_rate = sum(lang_bans.values()) / max(sum(lang_totals.values()), 1)
        print(f"   Overall BAN rate: {overall_ban_rate:.4f}")
        for lang in sorted(lang_totals.keys()):
            ban_rate = lang_bans.get(lang, 0) / max(lang_totals[lang], 1)
            ratio = ban_rate / max(overall_ban_rate, 1e-8)
            flag = "DISPARITY" if ratio > 1.5 and ban_rate > 0 else ""
            print(f"   {lang:6s}: BAN rate={ban_rate:.4f} (ratio={ratio:.2f}) {flag}")


def test_escalation_scenarios(model_path: str):
    """
    Test specific behavioral scenarios to verify escalation logic.
    Uses synthetic observations to control exact inputs.
    """
    model = MaskablePPO.load(model_path)

    print(f"\n{'='*60}")
    print(f"ESCALATION SCENARIO TESTS")
    print(f"{'='*60}")

    # Scenario 1: Gradual escalation with repeat offenses
    print(f"\n--- Scenario 1: Repeat Mild Offender ---")
    embedding = np.random.randn(384).astype(np.float32) * 0.1
    
    warns = 0.0
    timeouts = 0.0
    total = 0.0
    
    for step, tox in enumerate([0.1, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]):
        obs = {
            "message_embedding": embedding,
            "toxicity_score": np.array([tox], dtype=np.float32),
            "user_history": np.array([
                min(warns / 5.0, 1.0),
                min(timeouts / 3.0, 1.0),
                min(total / 10.0, 1.0),
            ], dtype=np.float32),
            "server_heat": np.array([0.2, 0.1], dtype=np.float32),
            "language_id": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32),  # English
        }
        
        mask = np.ones(5, dtype=bool)
        if tox < 0.20:
            mask[3:] = False
        elif tox < 0.35:
            mask[4] = False
        elif tox >= 0.90:
            mask[0] = False
        
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)
        
        if action == 1: warns += 1; total += 1
        elif action == 2: total += 1
        elif action == 3: timeouts += 1; total += 1
        elif action == 4: total += 1
        
        print(f"   Step {step+1}: tox={tox:.2f} → {ACTION_NAMES[action]} "
              f"(w={warns:.0f}, t={timeouts:.0f}, inf={total:.0f})")

    # Scenario 2: Rehabilitation
    print(f"\n--- Scenario 2: Rehabilitation (toxic then clean) ---")
    warns = 2.0
    timeouts = 1.0
    total = 3.0
    
    for step, tox in enumerate([0.05, 0.08, 0.03, 0.10, 0.05]):
        obs = {
            "message_embedding": embedding,
            "toxicity_score": np.array([tox], dtype=np.float32),
            "user_history": np.array([
                min(warns / 5.0, 1.0),
                min(timeouts / 3.0, 1.0),
                min(total / 10.0, 1.0),
            ], dtype=np.float32),
            "server_heat": np.array([0.1, 0.05], dtype=np.float32),
            "language_id": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32),  # English
        }
        
        mask = np.ones(5, dtype=bool)
        mask[3:] = False  # Low tox
        
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        print(f"   Step {step+1}: tox={tox:.2f} → {ACTION_NAMES[int(action)]}")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/models/best/best_model.zip"
    evaluate_on_episodes(model_path, n_episodes=100)
    test_escalation_scenarios(model_path)
