"""
Day 31 — Baseline Comparison.
Day 34 — Confusion Matrix (5×5 across ALLOW/WARN/DELETE/TIMEOUT/BAN).
Day 35 — Fairness Audit (per-language BAN rate vs English baseline).

Three baselines compared against the trained RL agent:
  Baseline A: Keyword regex filter (fails on nuance/context)
  Baseline B: Static threshold — DELETE if XLM-R > 0.80 (fails on escalation)
  RL Agent:   MaskablePPO with full state

RUN:
  python -m src.diagnostics.baseline_comparison
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import re
from collections import defaultdict
from sb3_contrib import MaskablePPO
from src.env.discord_env import DiscordEnv, LANG_TO_IDX, NUM_LANGUAGES

ACTION_NAMES = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]


# ═══════════════════════════════════════════════════════════════
# BASELINE A: Keyword Regex Filter
# ═══════════════════════════════════════════════════════════════

TOXIC_KEYWORDS = re.compile(
    r"\b(idiot|stupid|garbage|trash|moron|loser|pathetic|useless|"
    r"shut up|stfu|kill|die|hate|worst|terrible|awful|braindead|dumb|"
    r"noob|uninstall|kys)\b",
    re.IGNORECASE
)

def baseline_keyword(text: str) -> int:
    """Baseline A: DELETE(2) if keywords found, else ALLOW(0)."""
    if TOXIC_KEYWORDS.search(text):
        return 2  # DELETE
    return 0  # ALLOW


# ═══════════════════════════════════════════════════════════════
# BASELINE B: Static Threshold on XLM-R Score
# ═══════════════════════════════════════════════════════════════

def baseline_threshold(tox_score: float) -> int:
    """Baseline B: DELETE(2) if score > 0.80, WARN(1) if > 0.50, else ALLOW(0)."""
    if tox_score > 0.80:
        return 2  # DELETE
    elif tox_score > 0.50:
        return 1  # WARN
    return 0  # ALLOW


# ═══════════════════════════════════════════════════════════════
# RL AGENT
# ═══════════════════════════════════════════════════════════════

class RLAgentEvaluator:
    def __init__(self, model_path: str = "data/models/best/best_model.zip"):
        self.model = MaskablePPO.load(model_path)
        self.ledger = {}

    def predict(self, embedding, tox, user_id, lang):
        if user_id not in self.ledger:
            self.ledger[user_id] = {"warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0}
        led = self.ledger[user_id]

        lang_vec = np.zeros(NUM_LANGUAGES, dtype=np.float32)
        li = LANG_TO_IDX.get(lang, -1)
        if li >= 0:
            lang_vec[li] = 1.0

        obs = {
            "message_embedding": embedding,
            "toxicity_score": np.array([np.clip(tox, 0.0, 1.0)], dtype=np.float32),
            "user_history": np.array([
                min(led["warns"] / 5.0, 1.0),
                min(led["timeouts"] / 3.0, 1.0),
                min(led["total_infractions"] / 10.0, 1.0),
            ], dtype=np.float32),
            "server_heat": np.array([0.0, 0.0], dtype=np.float32),
            "language_id": lang_vec,
        }

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

        action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)

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

        return action


# ═══════════════════════════════════════════════════════════════
# EVALUATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════

def compute_best_action(tox, total_inf, timeouts):
    """Ground truth based on reward function logic."""
    if tox < 0.30:
        return 0
    if total_inf < 1:
        return 1
    if timeouts == 0:
        if total_inf >= 2:
            return 3
        elif tox >= 0.80:
            return 2
        else:
            return 1
    if timeouts >= 2:
        return 4 if tox >= 0.70 else 3
    return 3


def run_baseline_comparison():
    data_dir = "data/processed"

    with open(f"{data_dir}/episodes.json", "r") as f:
        episodes = json.load(f)
    with open(f"{data_dir}/context_strings.json", "r", encoding="utf-8") as f:
        context_strings = json.load(f)

    embeddings = np.load(f"{data_dir}/context_embeddings.npy")
    toxicity_scores = np.load(f"{data_dir}/toxicity_scores.npy")

    rl_agent = RLAgentEvaluator()

    # Metrics per system
    systems = {
        "Keyword Filter": {"actions": [], "fp": 0, "fn": 0, "correct": 0},
        "Static Threshold": {"actions": [], "fp": 0, "fn": 0, "correct": 0},
        "RL Agent": {"actions": [], "fp": 0, "fn": 0, "correct": 0},
    }

    # Day 34: Confusion matrices (5×5)
    confusion = {name: np.zeros((5, 5), dtype=int) for name in systems}

    # Day 35: Per-language tracking
    lang_actions = {name: defaultdict(lambda: defaultdict(int)) for name in systems}

    total_steps = 0

    # Build raw text lookup
    raw_texts = {}
    for entry in context_strings:
        raw_texts[entry["global_step_index"]] = entry.get("raw_text", "")

    print(f"Evaluating {len(episodes)} episodes...")

    for ep in episodes:
        step_indices = ep["step_indices"]
        user_ids = ep["user_ids"]
        languages = ep.get("languages", ["en"] * len(step_indices))

        # Track per-episode state for ground truth
        gt_ledger = {}

        for i, idx in enumerate(step_indices):
            tox = float(toxicity_scores[idx])
            uid = user_ids[i]
            lang = languages[i] if i < len(languages) else "en"
            text = raw_texts.get(idx, "")
            emb = np.nan_to_num(embeddings[idx].astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
            emb = np.clip(emb, -1.0, 1.0)

            # Ground truth
            if uid not in gt_ledger:
                gt_ledger[uid] = {"total_inf": 0.0, "timeouts": 0.0}
            gt = gt_ledger[uid]
            best = compute_best_action(tox, gt["total_inf"], gt["timeouts"])

            # Update ground truth ledger
            if best >= 1:
                gt["total_inf"] += 1.0
            if best == 3:
                gt["timeouts"] += 1.0

            # Evaluate each system
            predictions = {
                "Keyword Filter": baseline_keyword(text),
                "Static Threshold": baseline_threshold(tox),
                "RL Agent": rl_agent.predict(emb, tox, f"{ep['thread_id']}_{uid}", lang),
            }

            for name, pred in predictions.items():
                systems[name]["actions"].append(pred)
                confusion[name][best][pred] += 1
                lang_actions[name][lang][pred] += 1

                if pred == best:
                    systems[name]["correct"] += 1
                if tox < 0.30 and pred > 0:
                    systems[name]["fp"] += 1
                if tox >= 0.70 and pred == 0:
                    systems[name]["fn"] += 1

            total_steps += 1

    # ═══════════════════════════════════════════════════════════
    # Day 31: BASELINE COMPARISON REPORT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print(f"📊 DAY 31 — BASELINE COMPARISON ({total_steps} steps)")
    print(f"{'=' * 70}")

    print(f"\n{'System':20s} {'Accuracy':>10s} {'FP Rate':>10s} {'FN Rate':>10s}")
    print(f"{'─' * 52}")
    for name in systems:
        s = systems[name]
        acc = s["correct"] / max(total_steps, 1)
        fpr = s["fp"] / max(total_steps, 1)
        fnr = s["fn"] / max(total_steps, 1)
        print(f"{name:20s} {acc:10.4f} {fpr:10.4f} {fnr:10.4f}")

    # Action distributions
    print(f"\n   Action Distributions:")
    for name in systems:
        counts = defaultdict(int)
        for a in systems[name]["actions"]:
            counts[a] += 1
        dist_str = "  ".join(f"{ACTION_NAMES[a]}={counts.get(a,0)}" for a in range(5))
        print(f"     {name:20s}: {dist_str}")

    # ═══════════════════════════════════════════════════════════
    # Day 34: CONFUSION MATRIX
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print(f"📊 DAY 34 — CONFUSION MATRICES (5×5)")
    print(f"{'=' * 70}")

    for name in systems:
        cm = confusion[name]
        print(f"\n  {name}:")
        print(f"  {'':12s}", end="")
        for a in range(5):
            print(f" {ACTION_NAMES[a]:>8s}", end="")
        print(f"  {'(predicted →)':>14s}")

        for gt_a in range(5):
            print(f"  {ACTION_NAMES[gt_a]:12s}", end="")
            for pred_a in range(5):
                val = cm[gt_a][pred_a]
                print(f" {val:8d}", end="")
            row_total = cm[gt_a].sum()
            if row_total > 0:
                diag_pct = cm[gt_a][gt_a] / row_total * 100
                print(f"  ({diag_pct:5.1f}% correct)")
            else:
                print()
        print(f"  (actual ↓)")

    # ═══════════════════════════════════════════════════════════
    # Day 35: FAIRNESS AUDIT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print(f"📊 DAY 35 — FAIRNESS AUDIT (per-language BAN rates)")
    print(f"{'=' * 70}")

    for name in systems:
        la = lang_actions[name]
        print(f"\n  {name}:")

        # Compute overall BAN rate
        total_bans = sum(la[l].get(4, 0) for l in la)
        total_msgs = sum(sum(la[l].values()) for l in la)
        overall_ban_rate = total_bans / max(total_msgs, 1)

        # English baseline
        en_total = sum(la["en"].values()) if "en" in la else 1
        en_bans = la["en"].get(4, 0) if "en" in la else 0
        en_ban_rate = en_bans / max(en_total, 1)

        print(f"     Overall BAN rate: {overall_ban_rate:.4f} ({total_bans}/{total_msgs})")
        print(f"     English baseline: {en_ban_rate:.4f} ({en_bans}/{en_total})")

        disparity_found = False
        print(f"\n     {'Language':12s} {'Messages':>10s} {'BANs':>6s} {'BAN Rate':>10s} {'Ratio':>8s} {'Status'}")
        print(f"     {'─' * 62}")

        for lang in sorted(la.keys()):
            l_total = sum(la[lang].values())
            l_bans = la[lang].get(4, 0)
            l_rate = l_bans / max(l_total, 1)
            ratio = l_rate / max(overall_ban_rate, 1e-8)
            flag = "⚠️  DISPARITY" if ratio > 1.5 and l_bans > 0 else "✅"
            if ratio > 1.5 and l_bans > 0:
                disparity_found = True
            print(f"     {lang:12s} {l_total:10d} {l_bans:6d} {l_rate:10.4f} {ratio:8.2f} {flag}")

        if disparity_found:
            print(f"\n     ❌ FAIRNESS AUDIT FAILED — disparate impact detected")
            print(f"        Remediation: Retrain with higher λ initialization")
        else:
            print(f"\n     ✅ FAIRNESS AUDIT PASSED — no language exceeds 1.5× baseline")


if __name__ == "__main__":
    run_baseline_comparison()
