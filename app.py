"""
Multilingual ModBot — Shared Discord Server Simulator (Gradio demo for HF Spaces).

All connected visitors share ONE channel and ONE moderator ledger — like a real
Discord server. Pick a persona, send a message, and everyone else's view updates
within ~1 second via a Gradio Timer poll. Mod actions (warn, delete, timeout,
ban) are visible to all visitors. The persona radio is just a "speaking as"
selector — multiple visitors can speak as the same persona (their messages
stack on the same ledger, like alt-tabbed Discord sessions).
"""
import os
import sys
import threading
import time
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


# ── shared global state ────────────────────────────────────────────
CHANNEL: list = []        # list of message dicts, shared by all visitors
CHANNEL_LOCK = threading.Lock()
CHANNEL_ID = "global_channel"   # one logical channel for the whole server


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
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_channel() -> str:
    header = (
        '<div style="background:#2F3136; color:#FFF; padding:8px 14px; '
        'border-radius:8px 8px 0 0; font-weight:bold; font-family:sans-serif;">'
        '#general &middot; <span style="color:#3BA55C;">live · shared by all visitors</span>'
        '</div>'
    )
    with CHANNEL_LOCK:
        snapshot = list(CHANNEL)

    if not snapshot:
        body = (
            '<div style="background:#36393F; color:#888; padding:40px; '
            'text-align:center; border-radius:0 0 8px 8px; font-family:sans-serif;">'
            'No messages yet — pick a persona and say something. '
            'Other visitors will see it in real time.'
            '</div>'
        )
        return header + body

    rows = [
        '<div id="modbot-channel" style="background:#36393F; color:#DCDDDE; '
        'padding:12px; border-radius:0 0 8px 8px; font-family:sans-serif; '
        'max-height:480px; overflow-y:auto;">'
    ]
    for m in snapshot:
        name = _esc(m["name"])
        color = m["color"]
        deleted = m["decision"] in ("DELETE", "TIMEOUT", "BAN", "REJECTED")
        avatar_letter = name[0].upper()
        msg_style = "text-decoration:line-through; color:#72767D;" if deleted else ""
        threat_str = f' · threat={_esc(m["threat"])}' if m["threat"] else ""
        ts = m.get("ts", "")

        badge = DECISION_BADGE.get(m["decision"], "")
        badge_html = (
            f'<div style="font-size:0.85em; margin-top:2px;">{badge}</div>'
            if badge else ""
        )
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
            f'{ts} · tox={m["tox"]:.2f} · {_esc(m["lang"])}{threat_str}'
            f'      </span>'
            f'    </div>'
            f'    <div style="{msg_style} word-wrap:break-word;">{_esc(m["content"])}</div>'
            f'    {badge_html}'
            f'  </div>'
            f'</div>'
        )
    rows.append("</div>")
    # Auto-scroll to bottom on every render (so new messages stay visible).
    rows.append(
        '<script>'
        '(function(){var el=document.getElementById("modbot-channel");'
        'if(el){el.scrollTop=el.scrollHeight;}})();'
        '</script>'
    )
    return header + "".join(rows)


def render_profile(persona_id: str) -> str:
    name, color = PERSONA_DICT.get(persona_id, ("?", "#72767D"))
    p = moderator.get_user_profile(persona_id)
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
def send(persona_id: str, message: str):
    """Append a message from `persona_id` to the global channel."""
    if not message.strip():
        return render_channel(), render_profile(persona_id), ""
    name, color = PERSONA_DICT[persona_id]
    result = moderator.moderate(message, persona_id, channel_id=CHANNEL_ID)

    entry = {
        "name": name,
        "color": color,
        "content": message,
        "decision": result["decision"],
        "tox": result.get("toxicity") or 0.0,
        "lang": result.get("language", "?"),
        "threat": result.get("threat_detected"),
        "ts": time.strftime("%H:%M:%S"),
    }
    with CHANNEL_LOCK:
        CHANNEL.append(entry)
    return render_channel(), render_profile(persona_id), ""


def reset_channel(persona_id: str):
    """Wipe the channel AND the moderator ledger for everyone."""
    with CHANNEL_LOCK:
        CHANNEL.clear()
    moderator.user_ledger.clear()
    moderator.banned_users.clear()
    moderator.recent_toxicity.clear()
    moderator.recent_actions.clear()
    return render_channel(), render_profile(persona_id)


def tick(persona_id: str):
    """Periodic refresh — re-renders channel + profile for every connected client."""
    return render_channel(), render_profile(persona_id)


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

with gr.Blocks(title="ModBot — Shared Discord Simulator", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # Multilingual ModBot — Shared Discord Server
        **Live, multi-user demo.** Open this page on multiple devices, each pick a different persona,
        and chat together. Everyone sees the same `#general` channel and the moderator's
        decisions in real time (~1 s refresh).
        Reset wipes the channel and ledger for **all** visitors.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            channel_view = gr.HTML(value=render_channel())
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
                reset_btn = gr.Button("Reset channel (everyone)", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=[msg_input], label="Try these")

        with gr.Column(scale=1):
            gr.Markdown("### Selected persona")
            profile_view = gr.HTML(value=render_profile("alice"))
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

    send_btn.click(send, [persona, msg_input], [channel_view, profile_view, msg_input])
    msg_input.submit(send, [persona, msg_input], [channel_view, profile_view, msg_input])
    persona.change(render_profile, persona, profile_view)
    reset_btn.click(reset_channel, persona, [channel_view, profile_view])

    timer = gr.Timer(value=1.0)
    timer.tick(tick, persona, [channel_view, profile_view])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
