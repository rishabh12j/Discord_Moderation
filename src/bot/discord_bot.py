"""
FairMod Discord Bot.

Entry point: python -m src.bot.discord_bot

Required environment variables:
  DISCORD_BOT_TOKEN   — bot token from Discord Developer Portal
  DISCORD_GUILD_ID    — target guild ID (integer as string); optional, defaults to "default"

The bot:
  1. Loads persisted user infraction state from SQLite on startup
  2. Calls ProductionModerator.moderate() on every non-bot, non-command message
  3. Executes the moderation decision: delete, warn, timeout, or ban
  4. Writes every decision to the audit log in SQLite
  5. Syncs in-memory ledger → DB every SYNC_INTERVAL messages and at shutdown

Commands (prefix: !)
  !modlogs [N]           — show last N (default 20) moderation actions in the guild
  !modprofile <@user>    — show the user's infraction profile
  !modreset <@user>      — manually reset a user's infraction ledger (requires Manage Guild)
  !modstatus             — show bot health (model loaded, ledger size, recent FP/FN estimate)
"""

import os
import sys
import asyncio
import datetime
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands, tasks

# Add project root to path so imports work when running as a module
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.production_moderator import ProductionModerator
from src.bot.user_ledger_db import UserLedgerDB


# ── Configuration ────────────────────────────────────────────────

BOT_TOKEN     = os.environ.get("DISCORD_BOT_TOKEN", "")
GUILD_ID_STR  = os.environ.get("DISCORD_GUILD_ID", "default")
MODEL_PATH    = os.environ.get("MODERATION_MODEL_PATH", "data/models/best/best_model.zip")
DB_PATH       = os.environ.get("MODERATION_DB_PATH", "data/bot/user_ledger.db")

# Number of messages processed before auto-syncing ledger to DB
SYNC_INTERVAL = 100

# Timeout duration per offense (seconds): 1st = 5 min, 2nd = 30 min, 3rd+ = 2 hr
TIMEOUT_DURATIONS = [300, 1800, 7200]

# Prefix for mod commands — must not collide with other bots in the server
COMMAND_PREFIX = "!"

ACTION_NAMES = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]

# Embed colour for each decision
EMBED_COLOURS = {
    "ALLOW":   discord.Colour.green(),
    "WARN":    discord.Colour.gold(),
    "DELETE":  discord.Colour.orange(),
    "TIMEOUT": discord.Colour.red(),
    "BAN":     discord.Colour.dark_red(),
    "REJECTED": discord.Colour.dark_grey(),
}

# ── Bot setup ────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True   # Required: Privileged Intent — enable in Dev Portal
intents.members = True           # Required for timeout/ban actions

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

moderator: Optional[ProductionModerator] = None
ledger_db: Optional[UserLedgerDB] = None
_messages_since_sync: int = 0


# ── Startup / shutdown ───────────────────────────────────────────

@bot.event
async def on_ready():
    global moderator, ledger_db

    print(f"[FairMod] Logged in as {bot.user} (id={bot.user.id})")

    # Load persistent DB
    ledger_db = UserLedgerDB(DB_PATH)
    print(f"[FairMod] SQLite ledger at {DB_PATH}")

    # Load moderation model (blocking — run in executor to avoid blocking event loop)
    loop = asyncio.get_event_loop()
    try:
        moderator = await loop.run_in_executor(
            None,
            lambda: ProductionModerator(model_path=MODEL_PATH)
        )
        print(f"[FairMod] Model loaded from {MODEL_PATH}")
    except Exception as exc:
        print(f"[FairMod] FATAL: Could not load model: {exc}")
        await bot.close()
        return

    # Restore user history from DB
    ledger_db.load_into_moderator(moderator, guild_id=GUILD_ID_STR)

    # Start periodic sync task
    periodic_sync.start()
    print("[FairMod] Ready.")


@bot.event
async def on_disconnect():
    _flush_ledger()


async def _shutdown():
    """Graceful shutdown: sync ledger then close."""
    _flush_ledger()
    await bot.close()


def _flush_ledger():
    if moderator is not None and ledger_db is not None:
        ledger_db.sync_from_moderator(moderator, guild_id=GUILD_ID_STR)
        print("[FairMod] Ledger synced to DB.")


@tasks.loop(minutes=5)
async def periodic_sync():
    """Flush in-memory ledger to DB every 5 minutes."""
    _flush_ledger()


# ── Core moderation handler ──────────────────────────────────────

@bot.event
async def on_message(message: discord.Message):
    global _messages_since_sync

    # Ignore bot messages and DMs
    if message.author.bot or message.guild is None:
        return

    # Let command processor handle command messages first
    await bot.process_commands(message)
    if message.content.startswith(COMMAND_PREFIX):
        return

    if moderator is None:
        return  # Still initialising

    content = message.content.strip()
    if not content:
        return

    user_id   = str(message.author.id)
    channel_id = str(message.channel.id)
    guild_id  = str(message.guild.id)

    # --- Run moderator in executor (CPU-bound: embedding + transformer inference) ---
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: moderator.moderate(content, user_id, channel_id)
    )

    decision    = result["decision"]
    toxicity    = result.get("toxicity") or 0.0
    language    = result.get("language", "?")
    threat_cat  = result.get("threat_detected")
    preview     = content[:200]

    # --- Log to DB ---
    if ledger_db is not None:
        ledger_db.log_action(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            decision=decision,
            toxicity=toxicity,
            language=language,
            threat_cat=threat_cat,
            message_preview=preview,
        )

    # --- Periodic ledger sync ---
    _messages_since_sync += 1
    if _messages_since_sync >= SYNC_INTERVAL:
        _messages_since_sync = 0
        _flush_ledger()

    # --- Execute enforcement action ---
    if decision == "ALLOW":
        return  # No action

    if decision == "REJECTED":
        # User is banned — delete any messages they somehow got through
        await _try_delete(message)
        return

    try:
        if decision == "WARN":
            await _action_warn(message, result)

        elif decision == "DELETE":
            await _action_delete(message, result)

        elif decision == "TIMEOUT":
            await _action_timeout(message, result)

        elif decision == "BAN":
            await _action_ban(message, result)

    except discord.Forbidden:
        print(f"[FairMod] Missing permissions to enforce {decision} on {message.author}")
    except discord.HTTPException as exc:
        print(f"[FairMod] HTTP error during {decision}: {exc}")


# ── Enforcement helpers ──────────────────────────────────────────

async def _try_delete(message: discord.Message):
    """Delete a message, silently ignoring 'already deleted' errors."""
    try:
        await message.delete()
    except (discord.NotFound, discord.Forbidden):
        pass


async def _action_warn(message: discord.Message, result: dict):
    """Reply with a public warning embed, do not delete the message."""
    warns = int(result.get("user_warns", 1))
    embed = discord.Embed(
        title="Warning",
        description=(
            f"{message.author.mention}, this message has been flagged.\n"
            f"Please keep the conversation respectful.\n\n"
            f"**Warning {warns} of 5** — repeated violations will result in "
            f"escalating consequences."
        ),
        colour=EMBED_COLOURS["WARN"],
        timestamp=datetime.datetime.utcnow(),
    )
    embed.set_footer(text=f"tox={result['toxicity']:.3f}  lang={result['language']}")
    await message.reply(embed=embed, mention_author=True, delete_after=30)


async def _action_delete(message: discord.Message, result: dict):
    """Delete the message and send an ephemeral-style reply that auto-deletes."""
    await _try_delete(message)

    threat = result.get("threat_detected")
    reason = f"Threat detected ({threat})" if threat else "Harmful content"

    embed = discord.Embed(
        title="Message Removed",
        description=(
            f"{message.author.mention}, your message was removed.\n"
            f"**Reason:** {reason}\n\n"
            f"Further violations may result in a timeout."
        ),
        colour=EMBED_COLOURS["DELETE"],
        timestamp=datetime.datetime.utcnow(),
    )
    embed.set_footer(text=f"tox={result['toxicity']:.3f}  lang={result['language']}")
    await message.channel.send(embed=embed, delete_after=20)


async def _action_timeout(message: discord.Message, result: dict):
    """Delete message and temporarily mute the user."""
    await _try_delete(message)

    timeouts = int(result.get("user_timeouts", 1))
    duration_sec = TIMEOUT_DURATIONS[min(timeouts - 1, len(TIMEOUT_DURATIONS) - 1)]
    duration = datetime.timedelta(seconds=duration_sec)
    duration_label = _format_duration(duration_sec)

    try:
        await message.author.timeout(duration, reason="FairMod: repeated violations")
    except (discord.Forbidden, AttributeError):
        pass  # Bot lacks Moderate Members permission or user not a Member

    embed = discord.Embed(
        title="⏱Timeout",
        description=(
            f"{message.author.mention} has been timed out for **{duration_label}**.\n"
            f"This is timeout **{timeouts}**.\n\n"
            f"Please review the server rules before returning."
        ),
        colour=EMBED_COLOURS["TIMEOUT"],
        timestamp=datetime.datetime.utcnow(),
    )
    embed.add_field(name="Total infractions", value=str(int(result.get("total_infractions", 0))))
    embed.set_footer(text=f"tox={result['toxicity']:.3f}  lang={result['language']}")
    await message.channel.send(embed=embed)


async def _action_ban(message: discord.Message, result: dict):
    """Delete message and ban the user from the server."""
    await _try_delete(message)

    threat = result.get("threat_detected")
    reason = f"FairMod: extreme violation — {threat}" if threat else "FairMod: extreme violation"

    try:
        await message.author.ban(reason=reason, delete_message_days=1)
    except (discord.Forbidden, AttributeError):
        pass

    embed = discord.Embed(
        title="🔨 User Banned",
        description=(
            f"{message.author.mention} has been permanently banned.\n"
            f"**Reason:** {reason}"
        ),
        colour=EMBED_COLOURS["BAN"],
        timestamp=datetime.datetime.utcnow(),
    )
    embed.add_field(name="Total infractions", value=str(int(result.get("total_infractions", 0))))
    embed.set_footer(text=f"tox={result['toxicity']:.3f}  lang={result['language']}")
    await message.channel.send(embed=embed)


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


# ── Mod commands ─────────────────────────────────────────────────

@bot.command(name="modlogs")
@commands.has_permissions(manage_messages=True)
async def cmd_modlogs(ctx: commands.Context, limit: int = 20):
    """Show the most recent moderation actions for this guild."""
    if ledger_db is None:
        await ctx.send("Ledger not loaded yet.")
        return

    limit = max(1, min(limit, 50))
    rows = ledger_db.get_recent_actions(guild_id=str(ctx.guild.id), limit=limit)

    if not rows:
        await ctx.send("No moderation actions recorded yet.")
        return

    lines = [f"**Last {len(rows)} actions in {ctx.guild.name}:**\n"]
    for r in rows:
        uid = r["user_id"]
        try:
            member = ctx.guild.get_member(int(uid)) or await ctx.guild.fetch_member(int(uid))
            name = member.display_name
        except Exception:
            name = f"<@{uid}>"

        ts = r["created_at"][:16] if r["created_at"] else "?"
        preview = r.get("message_preview", "")[:40].replace("\n", " ")
        tox = r.get("toxicity", 0.0)
        lines.append(
            f"`{ts}` **{r['decision']}** — {name} — tox={tox:.3f} — `{preview}`"
        )

    # Discord max message length is 2000 chars; chunk if needed
    content = "\n".join(lines)
    for chunk in _chunk_text(content, 1900):
        await ctx.send(chunk)


@bot.command(name="modprofile")
@commands.has_permissions(manage_messages=True)
async def cmd_modprofile(ctx: commands.Context, member: discord.Member = None):
    """Show a user's infraction profile."""
    if member is None:
        await ctx.send("Usage: `!modprofile @user`")
        return
    if moderator is None:
        await ctx.send("Moderator not loaded yet.")
        return

    profile = moderator.get_user_profile(str(member.id))
    status = profile.get("status", "clean")

    embed = discord.Embed(
        title=f"Moderation Profile — {member.display_name}",
        colour=discord.Colour.dark_red() if status == "banned" else discord.Colour.blurple(),
        timestamp=datetime.datetime.utcnow(),
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.add_field(name="Status",              value=status.upper(), inline=True)
    embed.add_field(name="Warns",               value=str(int(profile.get("warns", 0))),             inline=True)
    embed.add_field(name="Timeouts",            value=str(int(profile.get("timeouts", 0))),           inline=True)
    embed.add_field(name="Total infractions",   value=str(int(profile.get("total_infractions", 0))), inline=True)
    embed.add_field(name="Clean streak",        value=str(int(profile.get("clean_streak", 0))),      inline=True)

    # Recent history from DB
    if ledger_db is not None:
        history = ledger_db.get_user_history(str(member.id), guild_id=str(ctx.guild.id))
        if history:
            last5 = history[:5]
            lines = []
            for r in last5:
                ts = r["created_at"][:16] if r["created_at"] else "?"
                preview = r.get("message_preview", "")[:30].replace("\n", " ")
                lines.append(f"`{ts}` **{r['decision']}** — `{preview}`")
            embed.add_field(name="Recent actions", value="\n".join(lines), inline=False)

    await ctx.send(embed=embed)


@bot.command(name="modreset")
@commands.has_permissions(manage_guild=True)
async def cmd_modreset(ctx: commands.Context, member: discord.Member = None):
    """Manually reset a user's infraction ledger (Manage Guild required)."""
    if member is None:
        await ctx.send("Usage: `!modreset @user`")
        return
    if moderator is None:
        await ctx.send("Moderator not loaded yet.")
        return

    uid = str(member.id)
    from collections import deque

    moderator.user_ledger[uid] = {
        "warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0,
        "last_infraction_step": -1, "clean_streak": 0,
        "tox_momentum": deque(maxlen=moderator.MOMENTUM_WINDOW),
        "last_action": -1, "last_tox": 0.0,
    }
    moderator.banned_users.discard(uid)

    # Persist the reset
    if ledger_db is not None:
        ledger_db.save_user(uid, str(ctx.guild.id), {
            "warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0,
            "last_infraction_step": -1, "clean_streak": 0, "is_banned": False,
        })

    await ctx.send(
        f"Ledger for {member.mention} has been reset by {ctx.author.mention}.",
        allowed_mentions=discord.AllowedMentions(users=False),
    )


@bot.command(name="modstatus")
@commands.has_permissions(manage_messages=True)
async def cmd_modstatus(ctx: commands.Context):
    """Show bot health and statistics."""
    model_ok = moderator is not None
    db_ok    = ledger_db is not None

    ledger_size   = len(moderator.user_ledger) if model_ok else 0
    banned_count  = len(moderator.banned_users) if model_ok else 0

    embed = discord.Embed(
        title="FairMod Status",
        colour=discord.Colour.green() if model_ok and db_ok else discord.Colour.red(),
        timestamp=datetime.datetime.utcnow(),
    )
    embed.add_field(name="Model",       value="Loaded" if model_ok else "Not loaded", inline=True)
    embed.add_field(name="Database",    value="Connected" if db_ok else "Not connected", inline=True)
    embed.add_field(name="Guild ID",    value=GUILD_ID_STR, inline=True)
    embed.add_field(name="Users tracked", value=str(ledger_size), inline=True)
    embed.add_field(name="Banned users",  value=str(banned_count), inline=True)
    embed.add_field(name="Model path",    value=f"`{MODEL_PATH}`", inline=False)

    if model_ok:
        # Recent action distribution from in-memory buffer
        from collections import Counter
        recent = list(moderator.recent_actions)
        if recent:
            dist = Counter(recent)
            names = ACTION_NAMES
            dist_str = "  ".join(
                f"{names[a]}:{dist[a]}" for a in range(5) if dist.get(a, 0) > 0
            )
            embed.add_field(name="Last 50 actions", value=dist_str or "—", inline=False)

    await ctx.send(embed=embed)


# ── Error handler ─────────────────────────────────────────────────

@bot.event
async def on_command_error(ctx: commands.Context, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission to use that command.", delete_after=10)
    elif isinstance(error, commands.MemberNotFound):
        await ctx.send("Member not found. Mention them directly: `!modprofile @user`", delete_after=10)
    elif isinstance(error, commands.CommandNotFound):
        pass  # Silently ignore unknown commands
    else:
        print(f"[FairMod] Command error in {ctx.command}: {error}")


# ── Utility ──────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = 1900):
    """Split text into chunks that fit within Discord's message limit."""
    lines = text.split("\n")
    chunk, chunks = [], []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > size:
            chunks.append("\n".join(chunk))
            chunk, current_len = [], 0
        chunk.append(line)
        current_len += len(line) + 1
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks


# ── Entry point ──────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN environment variable not set.")
        print("  Export it before running:")
        print("  export DISCORD_BOT_TOKEN=your_token_here")
        sys.exit(1)

    try:
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure ledger is persisted even on Ctrl+C
        _flush_ledger()


if __name__ == "__main__":
    main()
