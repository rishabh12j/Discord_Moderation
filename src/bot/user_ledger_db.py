"""
SQLite Persistent User Ledger.

Replaces the in-memory dict in ProductionModerator so user infraction
history survives bot restarts. Schema mirrors the in-memory ledger exactly
so ProductionModerator can load/save without any logic changes.
"""
import sqlite3
import os
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_ledger (
    user_id             TEXT PRIMARY KEY,
    guild_id            TEXT NOT NULL DEFAULT 'default',
    warns               REAL NOT NULL DEFAULT 0.0,
    timeouts            REAL NOT NULL DEFAULT 0.0,
    total_infractions   REAL NOT NULL DEFAULT 0.0,
    last_infraction_step INTEGER NOT NULL DEFAULT -1,
    clean_streak        INTEGER NOT NULL DEFAULT 0,
    is_banned           INTEGER NOT NULL DEFAULT 0,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS moderation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    TEXT NOT NULL,
    channel_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    decision    TEXT NOT NULL,
    toxicity    REAL,
    language    TEXT,
    threat_cat  TEXT,
    message_preview TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class UserLedgerDB:
    """Thread-safe SQLite wrapper for persistent user infraction state."""

    def __init__(self, db_path: str = "data/bot/user_ledger.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ── user record helpers ─────────────────────────────────────

    def get_user(self, user_id: str, guild_id: str = "default") -> dict:
        """Return the ledger dict for a user, creating a fresh record if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_ledger WHERE user_id = ? AND guild_id = ?",
                (user_id, guild_id),
            ).fetchone()

        if row is None:
            return self._default_record(user_id, guild_id)
        return dict(row)

    def save_user(self, user_id: str, guild_id: str, record: dict):
        """Upsert a user ledger record."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO user_ledger
                    (user_id, guild_id, warns, timeouts, total_infractions,
                     last_infraction_step, clean_streak, is_banned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    warns               = excluded.warns,
                    timeouts            = excluded.timeouts,
                    total_infractions   = excluded.total_infractions,
                    last_infraction_step= excluded.last_infraction_step,
                    clean_streak        = excluded.clean_streak,
                    is_banned           = excluded.is_banned,
                    updated_at          = CURRENT_TIMESTAMP
            """, (
                user_id, guild_id,
                record.get("warns", 0.0),
                record.get("timeouts", 0.0),
                record.get("total_infractions", 0.0),
                record.get("last_infraction_step", -1),
                record.get("clean_streak", 0),
                int(record.get("is_banned", False)),
            ))

    def load_into_moderator(self, moderator, guild_id: str = "default"):
        """
        Bulk-load all user records from DB into a ProductionModerator instance.
        Call this once at bot startup.
        """
        from collections import deque
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM user_ledger WHERE guild_id = ?", (guild_id,)
            ).fetchall()

        loaded = 0
        for row in rows:
            uid = row["user_id"]
            moderator.user_ledger[uid] = {
                "warns":                float(row["warns"]),
                "timeouts":             float(row["timeouts"]),
                "total_infractions":    float(row["total_infractions"]),
                "last_infraction_step": int(row["last_infraction_step"]),
                "clean_streak":         int(row["clean_streak"]),
                "tox_momentum":         deque(maxlen=moderator.MOMENTUM_WINDOW),
                "last_action":          -1,
                "last_tox":             0.0,
            }
            if row["is_banned"]:
                moderator.banned_users.add(uid)
            loaded += 1

        print(f"[LedgerDB] Loaded {loaded} user records from {self.db_path}")

    def sync_from_moderator(self, moderator, guild_id: str = "default"):
        """
        Write all in-memory ledger state back to DB.
        Call periodically or at shutdown.
        """
        for uid, led in moderator.user_ledger.items():
            record = {**led, "is_banned": uid in moderator.banned_users}
            self.save_user(uid, guild_id, record)

    # ── moderation log ──────────────────────────────────────────

    def log_action(self, guild_id: str, channel_id: str, user_id: str,
                   decision: str, toxicity: float, language: str,
                   threat_cat: Optional[str], message_preview: str):
        """Record every moderation action for audit purposes."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO moderation_log
                    (guild_id, channel_id, user_id, decision, toxicity,
                     language, threat_cat, message_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (guild_id, channel_id, user_id, decision, toxicity,
                  language, threat_cat, message_preview[:200]))

    def get_recent_actions(self, guild_id: str, limit: int = 50) -> list:
        """Fetch the most recent moderation actions for a guild."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM moderation_log
                WHERE guild_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (guild_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_user_history(self, user_id: str, guild_id: str = "default") -> list:
        """Full moderation history for a specific user."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM moderation_log
                WHERE user_id = ? AND guild_id = ?
                ORDER BY created_at DESC LIMIT 100
            """, (user_id, guild_id)).fetchall()
        return [dict(r) for r in rows]

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _default_record(user_id: str, guild_id: str) -> dict:
        return {
            "user_id": user_id, "guild_id": guild_id,
            "warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0,
            "last_infraction_step": -1, "clean_streak": 0, "is_banned": 0,
        }
