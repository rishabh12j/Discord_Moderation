from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.episode_builder import load_base_data, episode_generator
from src.utils.build_user_norms import USER_NORMS_PATH  # adjust if path different
import json
from pathlib import Path
from src.utils.language_utils import SUPPORTED_LANGUAGES

@dataclass
class UserNorm:
    overall_avg_toxicity: float
    message_count: int


def load_user_norms(path: Path) -> Dict[str, UserNorm]:
    if not path.exists():
        raise FileNotFoundError(
            f"User norms file not found at {path}. Run build_user_norms.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    norms: Dict[str, UserNorm] = {}
    for user_id, data in raw.items():
        norms[user_id] = UserNorm(
            overall_avg_toxicity=float(data["overall_avg_toxicity"]),
            message_count=int(data["message_count"]),
        )
    return norms


class DiscordEnv(gym.Env):
    """
    Discord moderation environment with language-agnostic content features.

    Observation (Dict):
        - message_embedding: (384,) float32 in [-1, 1]
        - user_history: (2,) float32 in [0, 1]
            [overall_avg_tox, msg_count_norm]
        - server_stats: (4,) float32 in [0, 1]
            placeholder for recent server-wide metrics

    Action (Discrete(4)):
        0 = ALLOW
        1 = WARN
        2 = DELETE
        3 = BAN
    """

    metadata = {"render_modes": []}

    EMBEDDING_DIM = 384
    USER_HISTORY_DIM = 2
    SERVER_STATS_DIM = 4

    def __init__(
        self,
        chunk_size: int = 20,
        max_steps_per_chunk: Optional[int] = None,
    ):
        super().__init__()

        # ----- observation & action spaces -----
        self.observation_space = spaces.Dict(
            {
                "message_embedding": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.EMBEDDING_DIM,),
                    dtype=np.float32,
                ),
                "user_history": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.USER_HISTORY_DIM,),
                    dtype=np.float32,
                ),
                "server_stats": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.SERVER_STATS_DIM,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Discrete(4)

        # ----- data -----
        df, embeddings, scores = load_base_data()
        self._df = df
        self._embeddings = embeddings
        self._scores = scores

        self._episodes = list(
            episode_generator(df, embeddings, scores, chunk_size=chunk_size)
        )
        if not self._episodes:
            raise ValueError("No episode chunks generated. Check chunk_size and data.")

        self._num_episodes = len(self._episodes)
        self._chunk_size = chunk_size
        self._max_steps_per_chunk = max_steps_per_chunk or chunk_size

        # ----- user norms -----
        self._user_norms = load_user_norms(USER_NORMS_PATH)
        self._user_msg_count_stats = self._compute_msg_count_stats()

        # ----- state -----
        self._current_episode_index: int = 0
        self._current_step_in_chunk: int = 0
        self._current_chunk = None
        self._current_language: Optional[str] = None
        self._current_user_id: Optional[str] = None

        # server-level running stats (placeholder)
        self._server_recent_toxicity: float = 0.0
        self._server_ban_rate: float = 0.0
        self._server_warn_rate: float = 0.0
        self._server_engagement: float = 1.0

        self._total_actions: int = 0
        self._total_bans: int = 0
        self._total_warns: int = 0

        self._rng = np.random.default_rng()
        # Tracks language distribution per chunk for fairness auditing
        self._current_language_distribution: Counter = Counter()
        
        # History buffers to track actions and rewards for the current chunk
        self._episode_actions_history: List[int] = []
        self._episode_rewards_history: List[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_msg_count_stats(self) -> Dict[str, float]:
        counts = np.array(
            [u.message_count for u in self._user_norms.values()], dtype=np.float32
        )
        return {
            "min": float(counts.min()),
            "max": float(counts.max()),
            "mean": float(counts.mean()),
        }

    def _sample_episode(self):
        self._current_episode_index = int(self._rng.integers(0, self._num_episodes))
        self._current_chunk = self._episodes[self._current_episode_index]
        self._current_step_in_chunk = 0

    def _get_user_norm_features(self, user_id: str) -> np.ndarray:
        norm = self._user_norms.get(
            user_id,
            UserNorm(overall_avg_toxicity=0.0, message_count=1),
        )
        mc = float(norm.message_count)
        mc_min = self._user_msg_count_stats["min"]
        mc_max = self._user_msg_count_stats["max"]
        # simple min-max normalization, avoid divide-by-zero
        if mc_max > mc_min:
            msg_count_norm = (mc - mc_min) / (mc_max - mc_min)
        else:
            msg_count_norm = 0.0

        return np.array(
            [norm.overall_avg_toxicity, msg_count_norm],
            dtype=np.float32,
        )

    def _get_server_stats_vec(self) -> np.ndarray:
        return np.array(
            [
                self._server_recent_toxicity,
                self._server_ban_rate,
                self._server_warn_rate,
                self._server_engagement,
            ],
            dtype=np.float32,
        )

    def _build_observation(self, idx: int) -> Dict[str, np.ndarray]:
        emb = self._current_chunk.embeddings[idx].astype(np.float32)
        user_id = self._current_chunk.user_ids[idx]

        user_hist = self._get_user_norm_features(user_id)
        server_stats = self._get_server_stats_vec()

        # internal tracking only (not in observation)
        self._current_language = self._current_chunk.languages[idx]
        self._current_user_id = user_id

        return {
            "message_embedding": emb,
            "user_history": user_hist,
            "server_stats": server_stats,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment to a new episode chunk and clears histories.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a random episode chunk 
        self._sample_episode()
        
        # Reset step counter 
        self._current_step_in_chunk = 0

        # Reset episode histories
        self._episode_actions_history = []
        self._episode_rewards_history = []

        # Track current_language_distribution = Counter() for fairness auditing 
        self._current_language_distribution = Counter(self._current_chunk.languages)

        # Build the initial observation
        obs = self._build_observation(self._current_step_in_chunk)

        # Info dict can expose the internal language distribution for debugging/callbacks
        info = {
            "episode_language_distribution": dict(self._current_language_distribution),
            "starting_thread_id": self._current_chunk.thread_id
        }

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes the chosen action, updates the environment state, and returns the next observation.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        idx = self._current_step_in_chunk
        
        # 1. Record action in the episode history (added in Day 9)
        self._episode_actions_history.append(action)
        
        # 2. State Transition Logic (Day 11)
        # If DELETE (action == 2), scrub the message from the chunk's memory
        if action == 2:
            self._current_chunk.messages[idx] = "[DELETED]"
            # Set its embedding to a zero vector so the policy forgets it
            self._current_chunk.embeddings[idx] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            
        # 3. Advance the step counter
        self._current_step_in_chunk += 1
        
        # 4. Check for episode termination/truncation
        terminated = False
        truncated = False
        
        # If we reached the end of the chunk
        if self._current_step_in_chunk >= len(self._current_chunk.messages) or self._current_step_in_chunk >= self._max_steps_per_chunk:
            truncated = True
            # Build observation using the last valid index to prevent out-of-bounds errors
            obs = self._build_observation(idx) 
        else:
            # Normal observation for the next step
            obs = self._build_observation(self._current_step_in_chunk)
            
        # 5. Reward (Placeholder for Phase 3)
        reward = 0.0
        self._episode_rewards_history.append(reward)
        
        # 6. Server Stats Bookkeeping (from Day 8)
        self._total_actions += 1
        if action == 1:
            self._total_warns += 1
        elif action == 3:
            self._total_bans += 1
            
        if self._total_actions > 0:
            self._server_ban_rate = self._total_bans / self._total_actions
            self._server_warn_rate = self._total_warns / self._total_actions
            
        info = {}
        
        return obs, reward, terminated, truncated, info

    # Optional: stub for MaskablePPO (to be filled on Day 10)
    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions for the current state.
        Shape: (4,) mapping to [ALLOW, WARN, DELETE, BAN].
        True means the action is allowed.
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        
        # Get current step data
        idx = self._current_step_in_chunk
        current_text = self._current_chunk.messages[idx]
        current_score = self._current_chunk.toxicity_scores[idx]
        current_lang = self._current_language
        
        # Rule 1: Admin protection (Mocked for synthetic data)
        # We will designate 'user_0000' as our mock admin for this environment.
        if self._current_user_id == "user_0000":
            mask[1] = False  # No WARN
            mask[2] = False  # No DELETE
            mask[3] = False  # No BAN
            
        # Rule 2: Deleted message masking
        # (State transitions in Day 11 will replace text with "[DELETED]")
        if current_text == "[DELETED]":
            mask[2] = False  # Cannot DELETE again
            mask[3] = False  # Cannot BAN based on already deleted text
            
        # Rule 3: Multilingual Uncertainty Guardrail
        # Prevent irreversible BANs if the language isn't well-supported AND the score is highly ambiguous.
        if current_lang not in SUPPORTED_LANGUAGES and (0.4 < current_score < 0.6):
            mask[3] = False  # No BAN
            
        return mask
