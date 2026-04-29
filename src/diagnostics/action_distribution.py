"""
Diagnostic 2: Action Distribution Analysis

PURPOSE: See what the trained agent ACTUALLY does across many episodes.
If TIMEOUT (action 3) is almost never selected, the agent hasn't 
learned when to use it.

WHAT TO LOOK FOR:
  - TIMEOUT % near 0 → agent never learned this action
  - DELETE % very high → agent is stuck on DELETE as "safe enough"
  - BAN % near 0 → ledger-aware masks are working (good) OR
    agent never reaches ban-eligible state (check escalation depth)

RUN:
  python -m src.diagnostics.action_distribution
"""
import numpy as np
import json
import gymnasium as gym
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.env.discord_env import DiscordEnv


ACTION_NAMES = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def analyze_action_distribution(
    model_path: str = "data/models/best/best_model.zip",
    n_episodes: int = 200,
):
    env = DiscordEnv()
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO.load(model_path)

    print("=" * 65)
    print(f"DIAGNOSTIC 2: ACTION DISTRIBUTION ({n_episodes} episodes)")
    print("=" * 65)

    # ── Counters ────────────────────────────────────────────────
    total_action_counts = {i: 0 for i in range(5)}
    total_steps = 0

    # Per-toxicity-tier action counts
    tox_tiers = {
        "safe (<0.30)":   {"range": (0.0, 0.30), "actions": {i: 0 for i in range(5)}, "total": 0},
        "mild (0.30-0.70)": {"range": (0.30, 0.70), "actions": {i: 0 for i in range(5)}, "total": 0},
        "high (0.70-0.90)": {"range": (0.70, 0.90), "actions": {i: 0 for i in range(5)}, "total": 0},
        "extreme (≥0.90)":  {"range": (0.90, 1.01), "actions": {i: 0 for i in range(5)}, "total": 0},
    }

    # Track escalation depth per episode
    max_action_per_episode = []    # highest action index reached
    timeout_episodes = 0           # episodes where TIMEOUT was used
    ban_episodes = 0               # episodes where BAN was used
    
    # Track action sequences per user within episodes
    escalation_chains = []         # list of action sequences for repeat offenders

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_actions = []
        ep_user_actions = defaultdict(list)  # user_id → [actions...]

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)

            # Get current state info before stepping
            step_idx = env.current_episode.step_indices[env.current_step_in_episode]
            tox = float(env.toxicity_scores[step_idx])
            uid = env.current_episode.user_ids[env.current_step_in_episode]

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            total_action_counts[action] += 1
            total_steps += 1
            ep_actions.append(action)
            ep_user_actions[uid].append((action, tox))

            # Categorize by toxicity tier
            for tier_name, tier_data in tox_tiers.items():
                lo, hi = tier_data["range"]
                if lo <= tox < hi:
                    tier_data["actions"][action] += 1
                    tier_data["total"] += 1
                    break

        max_action_per_episode.append(max(ep_actions) if ep_actions else 0)
        if 3 in ep_actions:
            timeout_episodes += 1
        if 4 in ep_actions:
            ban_episodes += 1

        # Record escalation chains for users who got 2+ actions
        for uid, action_list in ep_user_actions.items():
            punitive = [(a, t) for a, t in action_list if a > 0]
            if len(punitive) >= 2:
                escalation_chains.append(punitive)

    # ── Print Results ───────────────────────────────────────────

    print(f"\nOverall Action Distribution ({total_steps} total steps)")
    for a in range(5):
        count = total_action_counts[a]
        pct = count / max(total_steps, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"   {ACTION_NAMES[a]:8s}: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"\nActions by Toxicity Tier")
    for tier_name, tier_data in tox_tiers.items():
        t = tier_data["total"]
        if t == 0:
            print(f"\n   {tier_name}: 0 messages")
            continue
        print(f"\n   {tier_name} ({t} messages):")
        for a in range(5):
            count = tier_data["actions"][a]
            pct = count / t * 100
            bar = "█" * int(pct / 2)
            print(f"      {ACTION_NAMES[a]:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nEscalation Depth Per Episode")
    max_action_per_episode = np.array(max_action_per_episode)
    for a in range(5):
        count = (max_action_per_episode == a).sum()
        pct = count / n_episodes * 100
        print(f"   Max action = {ACTION_NAMES[a]:8s}: {count:4d} episodes ({pct:5.1f}%)")

    print(f"\n   Episodes with TIMEOUT: {timeout_episodes}/{n_episodes} ({timeout_episodes/n_episodes*100:.1f}%)")
    print(f"   Episodes with BAN:     {ban_episodes}/{n_episodes} ({ban_episodes/n_episodes*100:.1f}%)")

    # ── Escalation chain analysis ───────────────────────────────
    print(f"\n🔗 Escalation Chains (users with 2+ punitive actions in one episode)")
    print(f"   Total chains found: {len(escalation_chains)}")

    if escalation_chains:
        # Show most common patterns
        from collections import Counter
        patterns = Counter()
        for chain in escalation_chains:
            action_seq = tuple(a for a, t in chain)
            patterns[action_seq] += 1

        print(f"\n   Most common punitive action sequences:")
        for pattern, count in patterns.most_common(10):
            names = " → ".join(ACTION_NAMES[a] for a in pattern)
            print(f"      [{count:3d}x] {names}")

        # Check for "stuck on DELETE" pattern
        stuck_delete = sum(1 for chain in escalation_chains
                          if all(a == 2 for a, t in chain) and len(chain) >= 3)
        if stuck_delete > 0:
            print(f"\n   {stuck_delete} chains where agent did DELETE 3+ times in a row")
            print(f"      → This confirms the agent is 'stuck' on DELETE and not escalating to TIMEOUT")

    # ── Verdict ─────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"DIAGNOSIS")
    print(f"{'=' * 65}")

    issues = []

    timeout_pct = total_action_counts[3] / max(total_steps, 1) * 100
    if timeout_pct < 1.0:
        issues.append(
            f"TIMEOUT usage is {timeout_pct:.2f}% — agent almost never uses it.\n"
            f"      → Either training data lacks episodes long enough for\n"
            f"        escalation, or the reward gap between DELETE and TIMEOUT\n"
            f"        is too small for PPO to differentiate."
        )

    ban_pct = total_action_counts[4] / max(total_steps, 1) * 100
    if ban_pct < 0.5 and ban_episodes == 0:
        issues.append(
            f"BAN usage is {ban_pct:.2f}% — may be fine (ledger masks prevent premature bans)\n"
            f"      but also check if agent ever reaches ban-eligible state."
        )

    # Check if DELETE dominates in high-toxicity tier
    high_tier = tox_tiers["extreme (≥0.90)"]
    if high_tier["total"] > 0:
        delete_in_extreme = high_tier["actions"][2] / high_tier["total"] * 100
        if delete_in_extreme > 60:
            issues.append(
                f"DELETE accounts for {delete_in_extreme:.0f}% of extreme-tox actions.\n"
                f"      → Agent over-relies on DELETE even for very toxic content.\n"
                f"        Should be escalating to TIMEOUT/BAN for repeat offenders."
            )

    if not issues:
        print("   Action distribution looks healthy!")
    else:
        for issue in issues:
            print(f"\n   {issue}")

    print()


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/models/best/best_model.zip"
    analyze_action_distribution(model_path, n_episodes=200)
