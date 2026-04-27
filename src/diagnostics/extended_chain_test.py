"""
Diagnostic 3: Extended Harassment Chain Test

PURPOSE: Feed a long escalating sequence from a single user (12 messages)
to see if the agent ever reaches TIMEOUT and BAN, or gets stuck on DELETE.

This simulates the exact scenario you care about:
  User starts mildly annoying → gets worse → keeps going → should be banned

RUN:
  python -m src.diagnostics.extended_chain_test
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.production_moderator import ProductionModerator


def run_extended_chain():
    moderator = ProductionModerator()
    user_id = "persistent_troll"

    # 12-message escalation: mild → moderate → severe → extreme
    # This chain should exercise the FULL escalation ladder
    test_messages = [
        # Phase 1: Mildly annoying (should → ALLOW or WARN)
        ("ugh this team is so bad",                        "mild annoyance"),
        ("can you guys even play this game",               "mild frustration"),
        
        # Phase 2: Getting personal (should → WARN → DELETE)
        ("you guys are literally the worst players ever",  "personal attack"),
        ("uninstall the game you're all garbage",          "direct insult"),
        ("trash team, absolute trash",                     "repeated insult"),
        
        # Phase 3: Escalating (should → DELETE → TIMEOUT)
        ("shut the hell up you useless noobs",             "aggressive"),
        ("you're all brain dead morons",                   "severe insult"),
        ("go kill yourselves",                             "extreme toxic"),
        
        # Phase 4: Continued after punishment (should → TIMEOUT → BAN)
        ("this whole server is cancer",                    "hateful"),
        ("I'll destroy every single one of you",           "veiled threat"),
        ("I will find where you live and end you",         "direct threat"),
        ("you're all dead, i swear to god",                "explicit threat"),
    ]

    print("=" * 70)
    print("🧪 DIAGNOSTIC 3: EXTENDED HARASSMENT CHAIN (12 messages, 1 user)")
    print("=" * 70)
    print()
    print("Expected escalation: ALLOW/WARN → WARN → DELETE → TIMEOUT → BAN")
    print()

    for i, (msg, desc) in enumerate(test_messages):
        result = moderator.moderate(msg, user_id)
        
        # Color-code the decision
        decision = result["decision"]
        tox = result.get("toxicity", "N/A")
        warns = result["user_warns"]
        timeouts = result["user_timeouts"]
        infractions = result["total_infractions"]
        
        # Visual severity indicator
        if decision == "ALLOW":
            icon = "🟢"
        elif decision == "WARN":
            icon = "🟡"
        elif decision == "DELETE":
            icon = "🟠"
        elif decision == "TIMEOUT":
            icon = "🔴"
        elif decision == "BAN":
            icon = "⛔"
        elif decision == "REJECTED":
            icon = "🚫"
        else:
            icon = "❓"

        print(f"  {icon} Step {i+1:2d} [{desc:>18s}]: '{msg}'")
        print(f"          → {decision:8s}  tox={tox}  "
              f"warns={warns:.0f}  timeouts={timeouts:.0f}  infractions={infractions:.0f}")
        
        if decision in ("BAN", "REJECTED"):
            remaining = len(test_messages) - i - 1
            if remaining > 0:
                print(f"\n  🚫 User banned. Remaining {remaining} messages will be auto-rejected.")
                # Still process remaining to show REJECTED behavior
                for j in range(i + 1, len(test_messages)):
                    msg_r, desc_r = test_messages[j]
                    result_r = moderator.moderate(msg_r, user_id)
                    print(f"  🚫 Step {j+1:2d} [{desc_r:>18s}]: → {result_r['decision']}")
            break

    # ── Summary ─────────────────────────────────────────────────
    led = moderator.user_ledger.get(user_id, {})
    profile = moderator.get_user_profile(user_id)
    
    print(f"\n{'=' * 70}")
    print(f"📋 FINAL USER PROFILE")
    print(f"{'=' * 70}")
    print(f"   Status:      {profile.get('status', 'unknown')}")
    print(f"   Warns:       {led.get('warns', 0):.0f}")
    print(f"   Timeouts:    {led.get('timeouts', 0):.0f}")
    print(f"   Infractions: {led.get('total_infractions', 0):.0f}")
    print(f"   Banned:      {user_id in moderator.banned_users}")
    
    print(f"\n🔍 DIAGNOSIS")
    warns = led.get("warns", 0)
    timeouts = led.get("timeouts", 0)
    is_banned = user_id in moderator.banned_users
    
    if is_banned and timeouts >= 1 and warns >= 1:
        print(f"   ✅ Full escalation chain completed: WARN({warns:.0f}) → TIMEOUT({timeouts:.0f}) → BAN")
    elif timeouts >= 1:
        print(f"   ⚠️  TIMEOUT was reached but BAN was not triggered.")
        print(f"      → Agent may need more training on the TIMEOUT→BAN transition.")
    elif warns >= 1 and timeouts == 0:
        print(f"   ❌ Agent never reached TIMEOUT (stuck on WARN/DELETE).")
        print(f"      → Confirms the Step 5 problem. Need training data fix or reward reshaping.")
    else:
        print(f"   ❌ Minimal escalation. Check if the model loaded correctly.")
    
    print()


def run_multi_user_test():
    """
    Bonus: Two users — one toxic, one innocent.
    Verify the agent doesn't punish the innocent user.
    """
    moderator = ProductionModerator()

    messages = [
        ("nice_user",  "hey everyone, fun game today!"),
        ("troll_user", "you're all garbage"),
        ("nice_user",  "good play on that last round"),
        ("troll_user", "shut up you worthless noobs"),
        ("nice_user",  "anyone want to group up after?"),
        ("troll_user", "i hope you all get hit by a bus"),
        ("nice_user",  "gg everyone, that was close"),
        ("troll_user", "i'll find where you live"),
    ]

    print("=" * 70)
    print("🧪 BONUS: MULTI-USER TEST (troll vs. innocent bystander)")
    print("=" * 70)
    print()

    for i, (uid, msg) in enumerate(messages):
        result = moderator.moderate(msg, uid)
        icon = {"ALLOW": "🟢", "WARN": "🟡", "DELETE": "🟠", 
                "TIMEOUT": "🔴", "BAN": "⛔", "REJECTED": "🚫"}.get(result["decision"], "❓")
        
        user_label = "👤 nice " if uid == "nice_user" else "💀 troll"
        print(f"  {icon} [{user_label}] '{msg}'")
        print(f"          → {result['decision']}  tox={result.get('toxicity', 'N/A')}  "
              f"inf={result['total_infractions']:.0f}")

    print(f"\n📋 Final Profiles:")
    for uid in ["nice_user", "troll_user"]:
        p = moderator.get_user_profile(uid)
        print(f"   {uid:12s}: {p}")

    nice_inf = moderator.user_ledger.get("nice_user", {}).get("total_infractions", 0)
    if nice_inf == 0:
        print(f"\n   ✅ Innocent user was never punished (0 infractions)")
    else:
        print(f"\n   ❌ Innocent user got {nice_inf:.0f} infraction(s) — false positive!")

    print()


if __name__ == "__main__":
    run_extended_chain()
    print("\n" + "─" * 70 + "\n")
    run_multi_user_test()
