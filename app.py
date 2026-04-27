"""
Multilingual ModBot — Discord Server Simulator (Gradio demo for HF Spaces).

Reviewers pick a persona (Alice / Bob / Diego / Priya / Yuki / Charlie),
send messages, and watch the moderation agent respond inline in a
Discord-styled chat feed. Each persona maintains its own infraction
ledger; switching personas updates the profile sidebar. Reset wipes the
channel for this session.
"""
import os
import sys
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent.production_moderator import ProductionModerator

print("Loading moderator (~30s on first launch)...")
moderator = ProductionModerator()
print("Ready.")


# ── personas ────────────────────────────────────────────────────────
PERSONAS = [
    ("alice",   "Alice",   "#5865F2"),
    ("bob",     "Bob",     "#3BA55C"),
    ("charlie", "Charlie", "#FAA61A"),
    ("diego",   "Diego",   "#ED4245"),
    ("priya",   "Priya",   "#EB459E"),
    ("yuki",    "Yuki",    "#9B59B6"),
]
PERSONA_DICT = {pid: (name, color) for pid, name, color in PERSONAS}


# ── rendering ───────────────────────────────────────────────────────
DECISION_BADGE = {
    "ALLOW":    "",
    "WARN":     '<span style="color:#FAA61A;">⚠️ Warned by ModBot</span>',
    "DELETE":   '<span style="color:#ED4245;">🗑️ Message removed by ModBot</span>',
    "TIMEOUT":  '<span style="color:#ED4245;">⏳ User timed out (10 min) by ModBot</span>',
    "BAN":      '<span style="color:#ED4245;">⛔ User banned by ModBot</span>',
    "REJECTED": '<span style="color:#888;">🚫 Message blocked (user already banned)</span>',
}


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_channel(messages: list) -> str:
    header = (
        '<div style="background:#2F3136; color:#FFF; padding:8px 14px; '
        'border-radius:8px 8px 0 0; font-weight:bold; font-family:sans-serif;">'
        '#general'
        '</div>'
    )
    if not messages:
        body = (
            '<div style="background:#36393F; color:#888; padding:40px; '
            'text-align:center; border-radius:0 0 8px 8px; font-family:sans-serif;">'
            'No messages yet — pick a persona and say something.'
            '</div>'
        )
        return header + body

    rows = [
        '<div style="background:#36393F; color:#DCDDDE; padding:12px; '
        'border-radius:0 0 8px 8px; font-family:sans-serif; '
        'max-height:480px; overflow-y:auto;">'
    ]
    for m in messages:
        name = _esc(m["name"])
        color = m["color"]
        deleted = m["decision"] in ("DELETE", "TIMEOUT", "BAN", "REJECTED")
        avatar_letter = name[0].upper()
        msg_style = "text-decoration:line-through; color:#72767D;" if deleted else ""
        threat_str = f' · threat={_esc(m["threat"])}' if m["threat"] else ""

        rows.append(
            f'<div style="display:flex; gap:12px; padding:8px 0; '
            f'border-bottom:1px solid #2F3136;">'
            f'  <div style="width:36px; height:36px; border-radius:50%; '
            f'background:{color}; color:white; display:flex; '
            f'align-items:center; justify-content:center; '
            f'font-weight:bold; flex-shrink:0;">{avatar_letter}</div>'
            f'  <div style="flex:1; min-width:0;">'
            f'    <div><span style="color:{color}; font-weight:600;">{name}</span> '
            f'      <span style="color:#72767D; font-size:0.8em;">'
            f'tox={m["tox"]:.2f} · {_esc(m["lang"])}{threat_str}'
            f'      </span>'
            f'    </div>'
            f'    <div style="{msg_style} word-wrap:break-word;">{_esc(m["content"])}</div>'
            f'    {"<div style=\"font-size:0.85em; margin-top:2px;\">" + DECISION_BADGE[m["decision"]] + "</div>" if DECISION_BADGE.get(m["decision"]) else ""}'
            f'  </div>'
            f'</div>'
        )
    rows.append('</div>')
    return header + ''.join(rows)


def render_profile(persona_id: str, scope: str) -> str:
    name, color = PERSONA_DICT.get(persona_id, ("?", "#72767D"))
    user_id = f"{scope}:{persona_id}"
    p = moderator.get_user_profile(user_id)
    rows = [
        ("Status", str(p.get("status", "—"))),
        ("Warns", f'{p.get("warns", 0):.0f}'),
        ("Timeouts", f'{p.get("timeouts", 0):.0f}'),
        ("Total infractions", f'{p.get("total_infractions", 0):.0f}'),
        ("Effective infractions", str(p.get("effective_infractions", 0))),
        ("Clean streak", f'{p.get("clean_streak", 0)}'),
    ]
    rows_html = "".join(
        f'<tr><td style="padding:4px 8px; color:#72767D;">{k}</td>'
        f'<td style="padding:4px 8px; text-align:right;"><b>{v}</b></td></tr>'
        for k, v in rows
    )
    return (
        f'<div style="border-left:4px solid {color}; padding:12px; '
        f'background:#2F3136; color:#DCDDDE; border-radius:4px; '
        f'font-family:sans-serif;">'
        f'<h3 style="margin:0 0 8px 0; color:{color};">{_esc(name)}</h3>'
        f'<table style="width:100%; border-collapse:collapse;">{rows_html}</table>'
        f'</div>'
    )


# ── handlers ────────────────────────────────────────────────────────
def send(channel: list, scope: str, persona_id: str, message: str):
    if not message.strip():
        return channel, render_channel(channel), render_profile(persona_id, scope), ""
    name, color = PERSONA_DICT[persona_id]
    user_id = f"{scope}:{persona_id}"
    result = moderator.moderate(message, user_id, channel_id=scope)

    channel = channel + [{
        "name": name,
        "color": color,
        "content": message,
        "decision": result["decision"],
        "tox": result.get("toxicity") or 0.0,
        "lang": result.get("language", "?"),
        "threat": result.get("threat_detected"),
    }]
    return channel, render_channel(channel), render_profile(persona_id, scope), ""


def reset_channel(scope: str, persona_id: str):
    for k in list(moderator.user_ledger.keys()):
        if k.startswith(f"{scope}:"):
            del moderator.user_ledger[k]
    moderator.banned_users = {
        u for u in moderator.banned_users if not u.startswith(f"{scope}:")
    }
    return [], render_channel([]), render_profile(persona_id, scope)


def switch_persona(persona_id: str, scope: str):
    return render_profile(persona_id, scope)


def new_scope():
    return uuid.uuid4().hex[:8]


# ── UI ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Good morning everyone, ready for the match!"],
    ["This game is kinda annoying today."],
    ["Are you guys seriously this bad?"],
    ["You are all actual garbage."],
    ["I will find where you live."],
    ["Just kidding guys, gg wp."],
    ["Bonjour tout le monde !"],
    ["क्या बकवास खेल रहे हो"],
    ["یا ریت تموت"],
    ["bhai theek se khelo na"],
]

with gr.Blocks(title="ModBot — Discord Simulator", theme=gr.themes.Base()) as demo:
    scope_state = gr.State(new_scope)
    channel_state = gr.State([])

    gr.Markdown(
        """
        # Multilingual ModBot — Discord Server Simulator
        Pick a persona, send a message, and watch the moderation agent respond in
        real time. Each persona keeps its own infraction history. Send 5+ benign
        messages from a previously-warned persona to see cool-down rehabilitation.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            channel_view = gr.HTML(value=render_channel([]))
            persona = gr.Radio(
                choices=[(name, pid) for pid, name, _ in PERSONAS],
                value="alice",
                label="Sending as",
            )
            msg_input = gr.Textbox(
                placeholder="Type a message and hit Send (or Enter)...",
                show_label=False,
                lines=1,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                reset_btn = gr.Button("Reset channel", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=[msg_input], label="Try these")

        with gr.Column(scale=1):
            gr.Markdown("### User profile")
            profile_view = gr.HTML(value=render_profile("alice", "init"))
            gr.Markdown(
                """
                **Action ladder**
                ```
                ALLOW → WARN → DELETE → TIMEOUT → BAN
                ```
                BAN requires a prior TIMEOUT.
                TIMEOUT requires 2+ prior infractions.
                After 5 clean messages, infractions decay.
                """
            )

    persona.change(switch_persona, [persona, scope_state], profile_view)
    send_btn.click(
        send,
        [channel_state, scope_state, persona, msg_input],
        [channel_state, channel_view, profile_view, msg_input],
    )
    msg_input.submit(
        send,
        [channel_state, scope_state, persona, msg_input],
        [channel_state, channel_view, profile_view, msg_input],
    )
    reset_btn.click(
        reset_channel,
        [scope_state, persona],
        [channel_state, channel_view, profile_view],
    )

    demo.load(switch_persona, [persona, scope_state], profile_view)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
