"""
Diagnostic 1: Episode Length Analysis

PURPOSE: Check if your training data has enough long episodes for the 
agent to learn multi-step escalation (WARN → DELETE → TIMEOUT → BAN).

If most episodes are 3-5 messages, the agent rarely sees a user accumulate
enough infractions within one episode to reach TIMEOUT/BAN territory.

WHAT TO LOOK FOR:
  - If median episode length < 6: the agent almost never practices 
    full escalation chains → need longer synthetic threads
  - If < 10% of episodes have 8+ messages: same problem
  - Check "toxic streak" analysis: how many episodes have 3+ consecutive
    toxic messages from the same user? That's the scenario where 
    TIMEOUT should fire.

RUN:
  python -m src.diagnostics.episode_analysis
"""
import json
import numpy as np
from collections import Counter, defaultdict


def analyze_episodes(
    episodes_file: str = "data/processed/episodes.json",
    toxicity_file: str = "data/processed/toxicity_scores.npy",
    context_file: str  = "data/processed/context_strings.json",
):
    # ── Load data ───────────────────────────────────────────────
    with open(episodes_file, "r") as f:
        episodes = json.load(f)

    tox_scores = np.load(toxicity_file)

    # Try to load context strings for raw text analysis
    context_data = None
    try:
        with open(context_file, "r", encoding="utf-8") as f:
            context_data = json.load(f)
    except FileNotFoundError:
        print("context_strings.json not found — skipping raw text analysis\n")

    print("=" * 65)
    print("DIAGNOSTIC 1: EPISODE LENGTH ANALYSIS")
    print("=" * 65)

    # ── 1. Basic episode length distribution ────────────────────
    lengths = [len(ep["step_indices"]) for ep in episodes]
    lengths = np.array(lengths)

    print(f"\n📏 Episode Length Distribution ({len(episodes)} episodes)")
    print(f"   Min:    {lengths.min()}")
    print(f"   Max:    {lengths.max()}")
    print(f"   Mean:   {lengths.mean():.1f}")
    print(f"   Median: {np.median(lengths):.0f}")
    print(f"   Std:    {lengths.std():.1f}")

    # Histogram
    bins = [(1, 3), (4, 5), (6, 7), (8, 10), (11, 15), (16, 100)]
    print(f"\n   Length Buckets:")
    for lo, hi in bins:
        count = np.sum((lengths >= lo) & (lengths <= hi))
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        print(f"   {label:>6s} msgs: {count:5d} ({pct:5.1f}%) {bar}")

    # ── 2. Toxic message distribution per episode ───────────────
    print(f"\n🔴 Toxicity Per Episode")

    toxic_per_episode = []
    episodes_with_escalation_potential = 0
    max_consecutive_toxic = []

    for ep in episodes:
        indices = ep["step_indices"]
        ep_tox = [float(tox_scores[i]) for i in indices]

        n_toxic = sum(1 for t in ep_tox if t > 0.50)
        toxic_per_episode.append(n_toxic)

        # Count max consecutive toxic messages (same-user streaks are what
        # matter for escalation, but consecutive toxic from anyone still helps)
        max_streak = 0
        current_streak = 0
        for t in ep_tox:
            if t > 0.50:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        max_consecutive_toxic.append(max_streak)

        # An episode has "escalation potential" if it has 3+ toxic messages
        if n_toxic >= 3:
            episodes_with_escalation_potential += 1

    toxic_per_episode = np.array(toxic_per_episode)
    max_consecutive_toxic = np.array(max_consecutive_toxic)

    print(f"   Avg toxic messages per episode:  {toxic_per_episode.mean():.1f}")
    print(f"   Episodes with 0 toxic msgs:      {(toxic_per_episode == 0).sum()} ({(toxic_per_episode == 0).mean()*100:.1f}%)")
    print(f"   Episodes with 1-2 toxic msgs:    {((toxic_per_episode >= 1) & (toxic_per_episode <= 2)).sum()}")
    print(f"   Episodes with 3+ toxic msgs:     {episodes_with_escalation_potential} ({episodes_with_escalation_potential/len(episodes)*100:.1f}%) ← escalation territory")
    print(f"   Episodes with 5+ toxic msgs:     {(toxic_per_episode >= 5).sum()}")

    print(f"\nConsecutive Toxic Streaks (max per episode)")
    print(f"   Avg longest streak:  {max_consecutive_toxic.mean():.1f}")
    print(f"   Streak = 0 (all safe):  {(max_consecutive_toxic == 0).sum()}")
    print(f"   Streak = 1 (isolated):  {(max_consecutive_toxic == 1).sum()}")
    print(f"   Streak = 2:             {(max_consecutive_toxic == 2).sum()}")
    print(f"   Streak ≥ 3 (escalation): {(max_consecutive_toxic >= 3).sum()} ({(max_consecutive_toxic >= 3).mean()*100:.1f}%) ← WARN→DELETE→TIMEOUT territory")
    print(f"   Streak ≥ 5 (full chain): {(max_consecutive_toxic >= 5).sum()} ({(max_consecutive_toxic >= 5).mean()*100:.1f}%) ← WARN→DELETE→TIMEOUT→BAN territory")

    # ── 3. Same-user toxic streak analysis ──────────────────────
    print(f"\nSame-User Toxic Streaks")
    print(f"   (How often does ONE user send 3+ toxic messages in a row?)")

    same_user_escalation_episodes = 0
    same_user_max_streaks = []

    for ep in episodes:
        indices = ep["step_indices"]
        user_ids = ep["user_ids"]
        ep_tox = [float(tox_scores[i]) for i in indices]

        # Track per-user consecutive toxic counts
        user_streak = defaultdict(int)
        user_max_streak = defaultdict(int)

        for j in range(len(indices)):
            uid = user_ids[j]
            if ep_tox[j] > 0.50:
                user_streak[uid] += 1
                user_max_streak[uid] = max(user_max_streak[uid], user_streak[uid])
            else:
                user_streak[uid] = 0

        if user_max_streak:
            best_streak = max(user_max_streak.values())
            same_user_max_streaks.append(best_streak)
            if best_streak >= 3:
                same_user_escalation_episodes += 1
        else:
            same_user_max_streaks.append(0)

    same_user_max_streaks = np.array(same_user_max_streaks)
    print(f"   Episodes where one user has 3+ toxic in a row: {same_user_escalation_episodes} ({same_user_escalation_episodes/len(episodes)*100:.1f}%)")
    print(f"   Episodes where one user has 5+ toxic in a row: {(same_user_max_streaks >= 5).sum()} ({(same_user_max_streaks >= 5).mean()*100:.1f}%)")

    # ── 4. Toxicity score distribution (bimodal check) ──────────
    print(f"\nToxicity Score Distribution (bimodal check)")
    all_tox = tox_scores
    bins_tox = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
                (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    for lo, hi in bins_tox:
        count = np.sum((all_tox >= lo) & (all_tox < hi))
        pct = count / len(all_tox) * 100
        bar = "█" * int(pct / 2)
        print(f"   [{lo:.2f}, {hi:.2f}): {count:6d} ({pct:5.1f}%) {bar}")

    # ── 5. Verdict ──────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"DIAGNOSIS")
    print(f"{'=' * 65}")

    issues = []

    if np.median(lengths) < 6:
        issues.append(
            f"Median episode length is {np.median(lengths):.0f} (need ≥ 6 for escalation).\n"
            f"      → Your synthetic threads are too short. The agent rarely sees\n"
            f"        a user accumulate enough infractions to reach TIMEOUT/BAN."
        )

    escalation_pct = episodes_with_escalation_potential / len(episodes) * 100
    if escalation_pct < 15:
        issues.append(
            f"Only {escalation_pct:.1f}% of episodes have 3+ toxic messages.\n"
            f"      → The agent has very few training examples of escalation.\n"
            f"        Need ≥ 20% for reliable TIMEOUT/BAN learning."
        )

    same_user_pct = same_user_escalation_episodes / len(episodes) * 100
    if same_user_pct < 10:
        issues.append(
            f"Only {same_user_pct:.1f}% of episodes have one user with 3+ toxic streak.\n"
            f"      → The escalation ladder depends on per-USER infractions, but\n"
            f"        most episodes spread toxicity across multiple users.\n"
            f"        Need episodes where ONE user is the persistent offender."
        )

    mid_range = np.sum((all_tox >= 0.3) & (all_tox < 0.7))
    mid_pct = mid_range / len(all_tox) * 100
    if mid_pct < 10:
        issues.append(
            f"Only {mid_pct:.1f}% of toxicity scores are in [0.3, 0.7] range.\n"
            f"      → Bimodal distribution confirmed. The classifier outputs\n"
            f"        almost exclusively ~0 or ~0.9+. Your escalation thresholds\n"
            f"        (0.30, 0.55, 0.85) only have two effective bins, not four.\n"
            f"        This isn't fatal but means toxicity level barely differentiates actions."
        )

    if not issues:
        print("   Training data looks healthy for escalation learning!")
    else:
        for issue in issues:
            print(f"\n   {issue}")

    print()


if __name__ == "__main__":
    analyze_episodes()
