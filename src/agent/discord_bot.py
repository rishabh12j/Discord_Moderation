"""
Discord bot wrapper for ProductionModerator.

Loads credentials from .env at the project root, listens to a single test
channel in a single guild, and translates moderator decisions into Discord
actions (warn reply, message delete, member timeout, member ban).

Run: python -m src.agent.discord_bot
"""
import os
import asyncio
from datetime import timedelta
from pathlib import Path

import discord
from discord import Intents

from src.agent.production_moderator import ProductionModerator


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f"Missing {env_path}. Create it with DISCORD_BOT_TOKEN, "
                                "DISCORD_GUILD_ID, DISCORD_TEST_CHANNEL_ID.")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(Path(__file__).resolve().parents[2] / ".env")

TOKEN = os.environ["DISCORD_BOT_TOKEN"]
GUILD_ID = int(os.environ["DISCORD_GUILD_ID"])
CHANNEL_ID = int(os.environ["DISCORD_TEST_CHANNEL_ID"])

TIMEOUT_DURATION = timedelta(minutes=10)

intents = Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)
moderator: ProductionModerator | None = None


@client.event
async def on_ready():
    global moderator
    print(f"Logged in as {client.user} (id={client.user.id})")
    guild = client.get_guild(GUILD_ID)
    channel = client.get_channel(CHANNEL_ID)
    print(f"Watching guild={guild} channel={channel}")
    if moderator is None:
        print("Loading ProductionModerator...")
        moderator = ProductionModerator()
        print("Moderator ready.")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.guild is None or message.guild.id != GUILD_ID:
        return
    if message.channel.id != CHANNEL_ID:
        return
    if moderator is None:
        return

    result = await asyncio.to_thread(
        moderator.moderate,
        message.content,
        str(message.author.id),
        str(message.channel.id),
    )
    decision = result["decision"]
    tox = result.get("toxicity")
    lang = result.get("language")
    threat = result.get("threat_detected")

    log = (f"[{decision}] user={message.author} tox={tox} lang={lang}"
           f"{f' threat={threat}' if threat else ''}: {message.content!r}")
    print(log)

    try:
        if decision == "ALLOW":
            return
        if decision == "WARN":
            await message.reply(
                f"⚠️ {message.author.mention} this message has been flagged. "
                f"Please keep the channel respectful.",
                mention_author=False,
            )
        elif decision == "DELETE":
            await message.delete()
            await message.channel.send(
                f"🗑️ A message from {message.author.mention} was removed for violating community guidelines."
            )
        elif decision == "TIMEOUT":
            await message.delete()
            await message.author.timeout(
                TIMEOUT_DURATION,
                reason=f"Auto-moderation: repeated infractions (tox={tox})",
            )
            await message.channel.send(
                f"⏳ {message.author.mention} has been timed out for "
                f"{int(TIMEOUT_DURATION.total_seconds() // 60)} minutes."
            )
        elif decision == "BAN":
            await message.delete()
            await message.guild.ban(
                message.author,
                reason=f"Auto-moderation: ban threshold reached (tox={tox})",
                delete_message_seconds=0,
            )
            await message.channel.send(
                f"⛔ {message.author} has been banned by the moderation agent."
            )
        elif decision == "REJECTED":
            await message.delete()
    except discord.Forbidden:
        print(f"  ! Forbidden: bot lacks permission to {decision} on {message.author}")
    except discord.HTTPException as e:
        print(f"  ! HTTPException during {decision}: {e}")


if __name__ == "__main__":
    client.run(TOKEN)
