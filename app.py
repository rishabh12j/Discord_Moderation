"""
Gradio demo for the multilingual moderation agent.

Deploys to Hugging Face Spaces. Reviewers can type a message, pick a user ID,
and see the decision, toxicity score, detected language, threat category,
and the full user ledger state — so the escalation ladder is visible.

Run locally:    python app.py
Deploy to HF:   push this repo to a Spaces repo with SDK=gradio.
"""
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent.production_moderator import ProductionModerator

print("Loading moderator (this takes ~30s on first launch)...")
moderator = ProductionModerator()
print("Ready.")

DECISION_EMOJI = {
    "ALLOW":    "✅",
    "WARN":     "⚠️",
    "DELETE":   "🗑️",
    "TIMEOUT":  "⏳",
    "BAN":      "⛔",
    "REJECTED": "🚫",
}


def moderate(message: str, user_id: str):
    if not message.strip():
        return "—", "", "", "", _profile_table(user_id)
    user_id = (user_id or "demo_user").strip() or "demo_user"
    result = moderator.moderate(message, user_id, channel_id="hf_demo")

    decision = f"{DECISION_EMOJI.get(result['decision'], '')} **{result['decision']}**"
    tox = f"{result.get('toxicity', 0):.3f}" if result.get("toxicity") is not None else "n/a"
    lang = result.get("language", "?")
    threat = result.get("threat_detected") or "—"

    return decision, tox, lang, threat, _profile_table(user_id)


def _profile_table(user_id: str) -> str:
    p = moderator.get_user_profile(user_id or "demo_user")
    rows = [
        ("Status", p.get("status", "—")),
        ("Warns", f"{p.get('warns', 0):.0f}"),
        ("Timeouts", f"{p.get('timeouts', 0):.0f}"),
        ("Total infractions", f"{p.get('total_infractions', 0):.0f}"),
        ("Effective infractions", f"{p.get('effective_infractions', 0)}"),
        ("Clean streak", f"{p.get('clean_streak', 0)}"),
    ]
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def reset_user(user_id: str):
    user_id = (user_id or "demo_user").strip() or "demo_user"
    moderator.user_ledger.pop(user_id, None)
    moderator.banned_users.discard(user_id)
    return "ledger cleared", "", "", "", _profile_table(user_id)


EXAMPLES = [
    ["Good morning everyone, ready for the match!", "demo_user"],
    ["This game is kinda annoying today.", "demo_user"],
    ["Are you guys seriously this bad?", "demo_user"],
    ["You are all actual garbage.", "demo_user"],
    ["I will find where you live.", "demo_user"],
    ["Bonjour tout le monde !", "user_fr"],
    ["क्या बकवास खेल रहे हो", "user_hi"],
    ["یا ریت تموت", "user_ar"],
    ["gg wp, that was close", "user_gamer"],
]

with gr.Blocks(title="Multilingual RL Moderation Agent") as demo:
    gr.Markdown(
        """
        # Multilingual RL Moderation Agent — Live Demo
        Constrained MaskablePPO over `ALLOW → WARN → DELETE → TIMEOUT → BAN`,
        13 languages, per-language calibration, Lagrangian fairness constraints.

        **How to use:** keep the same User ID and send a series of messages of
        increasing severity. Watch the escalation ladder respect masking
        (BAN requires a prior TIMEOUT, TIMEOUT requires prior infractions).
        Send 5+ benign messages to see the cool-down rehabilitation kick in.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            message = gr.Textbox(label="Message", placeholder="Type something...", lines=2)
            user_id = gr.Textbox(label="User ID", value="demo_user")
            with gr.Row():
                submit = gr.Button("Moderate", variant="primary")
                reset = gr.Button("Reset this user")
            gr.Examples(examples=EXAMPLES, inputs=[message, user_id], label="Try these")
        with gr.Column(scale=1):
            decision = gr.Markdown(label="Decision")
            with gr.Row():
                tox = gr.Textbox(label="Toxicity", interactive=False)
                lang = gr.Textbox(label="Language", interactive=False)
                threat = gr.Textbox(label="Threat", interactive=False)
            profile = gr.Markdown(
                value="| Field | Value |\n|---|---|\n",
                label="User profile",
            )

    submit.click(moderate, [message, user_id], [decision, tox, lang, threat, profile])
    message.submit(moderate, [message, user_id], [decision, tox, lang, threat, profile])
    reset.click(reset_user, [user_id], [decision, tox, lang, threat, profile])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
