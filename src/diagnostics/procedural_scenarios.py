"""
Day 30 — Procedural Scenario Generation.

Three specific 10+ message behavioral trajectories from the roadmap:

Scenario 1 — ESCALATION:
  Toxicity starts at 0.1, linearly increases to 0.6+.
  Tests: Does the agent issue preemptive warnings on momentum?

Scenario 2 — SUSTAINED TROLL:
  Messages static around 0.5 toxicity throughout.
  Tests: Does the agent escalate from WARN to TIMEOUT via recidivism?

Scenario 3 — REHABILITATION:
  One high-toxicity offense, then 9+ zero-toxicity messages.
  Tests: Does the ledger cool-down decay work? Does the agent
  eventually treat the user as clean again?

RUN:
  python -m src.diagnostics.procedural_scenarios
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.production_moderator import ProductionModerator

ACTION_ICONS = {
    "ALLOW": "🟢", "WARN": "🟡", "DELETE": "🟠",
    "TIMEOUT": "🔴", "BAN": "", "REJECTED": "🚫",
}


def print_step(i, msg, result, desc=""):
    icon = ACTION_ICONS.get(result["decision"], "❓")
    tox = result.get("toxicity", "N/A")
    tox_str = f"{tox:.3f}" if isinstance(tox, float) else str(tox)
    eff = result.get("effective_infractions", "?")
    clean = result.get("clean_streak", 0)
    print(f"  {icon} Step {i:2d}: [{desc:>14s}] '{msg[:55]}'"
          f"\n          → {result['decision']:8s}  tox={tox_str}  "
          f"warns={result['user_warns']:.0f}  timeouts={result['user_timeouts']:.0f}  "
          f"inf={result['total_infractions']:.0f}  eff_inf={eff}  clean={clean}")


# SCENARIO 1: ESCALATION — toxicity gradually increasing


def scenario_escalation(moderator: ProductionModerator):
    user_id = "escalation_user"

    # Simulate gradually worsening behavior
    messages = [
        ("hey everyone whats up",                     "friendly",   ),  # ~0.0
        ("this team is kinda slow today",             "mild gripe",  ),  # ~0.0-0.1
        ("seriously can anyone on this team aim",     "frustration", ),  # ~0.3-0.5
        ("you guys are playing like trash right now", "insulting",   ),  # ~0.6-0.8
        ("what a bunch of useless idiots",            "harsh",       ),  # ~0.8+
        ("you're all absolute garbage players",       "direct insult"),  # ~0.9+
        ("shut up you braindead morons",              "aggressive",  ),  # ~0.9+
        ("i hate every single one of you",            "hateful",     ),  # ~0.9+
        ("go uninstall the game losers",              "toxic",       ),  # ~0.9+
        ("you all deserve to lose",                   "hostile",     ),  # ~0.5-0.8
    ]

    print("=" * 70)
    print("SCENARIO 1: ESCALATION (gradual toxicity increase)")
    print("=" * 70)
    print("Expected: ALLOW early → WARN as toxicity rises → DELETE → TIMEOUT → BAN")
    print()

    for i, (msg, desc) in enumerate(messages, 1):
        result = moderator.moderate(msg, user_id)
        print_step(i, msg, result, desc)

    profile = moderator.get_user_profile(user_id)
    print(f"\n  Final: {profile}")

    # Evaluate
    total_inf = profile.get("total_infractions", 0)
    timeouts = profile.get("timeouts", 0)
    is_banned = user_id in moderator.banned_users

    if is_banned:
        print("  Full escalation → BAN reached")
    elif timeouts >= 1:
        print(f"  Reached TIMEOUT but not BAN (timeouts={timeouts})")
    else:
        print("  Insufficient escalation — never reached TIMEOUT")
    print()


# SCENARIO 2: SUSTAINED TROLL — persistent moderate toxicity


def scenario_sustained_troll(moderator: ProductionModerator):
    user_id = "sustained_troll"

    # All messages at roughly the same moderate toxicity level
    messages = [
        ("you guys are kinda bad",                "low troll",  ),
        ("terrible plays from this whole team",   "persistent", ),
        ("can nobody here play properly",         "persistent", ),
        ("this is the worst team I've ever seen", "persistent", ),
        ("honestly pathetic performance guys",    "persistent", ),
        ("you call yourselves gamers lol",        "persistent", ),
        ("embarrassing how bad you all are",      "persistent", ),
        ("i've seen bots play better than you",   "persistent", ),
        ("just quit already none of you are good","persistent", ),
        ("every single one of you is awful",      "persistent", ),
    ]

    print("=" * 70)
    print("SCENARIO 2: SUSTAINED TROLL (constant moderate toxicity)")
    print("=" * 70)
    print("Expected: WARN → DELETE → TIMEOUT through recidivism (not severity)")
    print()

    for i, (msg, desc) in enumerate(messages, 1):
        result = moderator.moderate(msg, user_id)
        print_step(i, msg, result, desc)

    profile = moderator.get_user_profile(user_id)
    print(f"\n  Final: {profile}")

    timeouts = profile.get("timeouts", 0)
    if timeouts >= 1:
        print("  Recidivism triggered TIMEOUT — agent detects sustained trolling")
    else:
        print("  Never reached TIMEOUT — agent not detecting sustained pattern")
    print()


# SCENARIO 3: REHABILITATION — one offense then clean behavior


def scenario_rehabilitation(moderator: ProductionModerator):
    user_id = "rehab_user"

    messages = [
        # The offense
        ("you are all complete garbage people",   "toxic",      ),

        # Recovery period — all clean
        ("sorry about that, bad day",             "apology",    ),
        ("nice play on that last round",          "friendly",   ),
        ("anyone want to group up?",              "friendly",   ),
        ("good game everyone",                    "friendly",   ),
        ("that was a really fun match",           "friendly",   ),
        ("gg wp, close one",                      "friendly",   ),
        ("haha that last fight was crazy",        "friendly",   ),
        ("you guys are really good at this",      "compliment", ),
        ("see you all tomorrow, great session",   "friendly",   ),
        ("hey everyone I'm back!",                "greeting",   ),
        ("let's have a good game today",          "positive",   ),

        # The test: one more mildly toxic message after rehabilitation
        # Should this be treated as first offense (decayed) or repeat?
        ("ugh this is so annoying",               "mild gripe", ),
    ]

    print("=" * 70)
    print("SCENARIO 3: REHABILITATION (offense + 10 clean messages)")
    print("=" * 70)
    print("Expected: WARN on offense → ALLOW through recovery →")
    print("          effective infractions decay → mild relapse treated leniently")
    print()

    for i, (msg, desc) in enumerate(messages, 1):
        result = moderator.moderate(msg, user_id)
        print_step(i, msg, result, desc)

        # Highlight cool-down progress
        if i > 1 and result["decision"] == "ALLOW" and i <= 12:
            eff = result.get("effective_infractions", "?")
            clean = result.get("clean_streak", 0)
            if clean == 5:
                print(f"          Cool-down threshold reached! Decay begins.")
            elif clean > 5 and isinstance(eff, (int, float)) and eff < 1:
                print(f"          Fully rehabilitated (eff_inf < 1)")

    profile = moderator.get_user_profile(user_id)
    print(f"\n  Final: {profile}")

    status = profile.get("status", "")
    if status == "rehabilitated":
        print("  User successfully rehabilitated — ledger decay works!")
    elif profile.get("effective_infractions", 99) < 1:
        print("  Effective infractions decayed below 1")
    else:
        print(f"  User status: {status} (eff_inf={profile.get('effective_infractions', '?')})")

    # Check: was the final mild relapse treated as a first offense?
    print()


if __name__ == "__main__":
    # Each scenario gets its own moderator instance for isolation
    print("\n" + "═" * 70)
    print("   DAY 30 — PROCEDURAL SCENARIO TESTS")
    print("═" * 70 + "\n")

    print("Loading models...\n")
    mod1 = ProductionModerator()

    scenario_escalation(mod1)

    print("─" * 70 + "\n")
    mod2 = ProductionModerator()
    scenario_sustained_troll(mod2)

    print("─" * 70 + "\n")
    mod3 = ProductionModerator()
    scenario_rehabilitation(mod3)

    print("═" * 70)
    print("   ALL SCENARIOS COMPLETE")
    print("═" * 70)
