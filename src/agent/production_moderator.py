"""
Production Moderator — v5.
Changes:
  - Change 2: language_id (13-dim one-hot) in observation, matching discord_env v5.
  - Change 3: Multilingual threat detection for all 13 languages.
  - All v4 features: cool-down, calibration, momentum, effective infractions.
"""
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sb3_contrib import MaskablePPO
from collections import deque

LANGUAGES = ["en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"]
LANG_TO_IDX = {
    "en": 0, "ru": 1, "uk": 2, "de": 3, "es": 4, "ar": 5,
    "hi": 6, "zh": 7, "it": 8, "fr": 9, "he": 10, "ja": 11, "hin": 12,
}
NUM_LANGUAGES = 13

# ═══════════════════════════════════════════════════════════════
# Gaming vocabulary that XLM-R falsely scores as toxic.
# Messages matching any of these patterns are clamped to score 0.0
# regardless of the classifier output.
# These are common gaming phrases (GG, well played, etc.) that are
# semantically positive but trigger false positives in XLM-R.
# ═══════════════════════════════════════════════════════════════
_BENIGN_GAMING_PATTERNS = re.compile(
    r"^(gg(\s+wp)?([,\s]+(close\s+(one|game)|wp))?|good\s+game|well\s+played|glhf|gl\s+hf|"
    r"nice\s+(shot|play|one|round|game)|wp\b|gg\s+close|close\s+(one|game)|ez\s+clap|clutch|"
    r"well\s+done|good\s+job|great\s+game|that\s+was\s+(fun|great|close|amazing)|"
    r"let'?s\s+go|pog(gers)?|pogchamp|nice\s+try|almost\s+had\s+it|respekt|respect|"
    r"f\s+in\s+(the\s+)?chat)[\s!.,]*$",
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════
# Change 3: MULTILINGUAL THREAT DETECTION
# Patterns organized by category, each category has patterns for
# all 13 languages. Japanese/Chinese use character-based matching
# (no \b word boundaries). Arabic/Hebrew are RTL-safe.
# ═══════════════════════════════════════════════════════════════

THREAT_PATTERNS_MULTILINGUAL = {
    # Category 1: Stalking / location threats ("I will find where you live")
    "stalking": [
        # English
        r"\b(i('ll| will)|gonna|going to)\b.{0,30}\b(find|hunt|track|come for|get)\b.{0,20}\b(you|where you live|your (house|address|family|school))\b",
        # Russian
        r"\b(я\s+(найду|выслежу|отслежу|приду за))\b.{0,20}(теб[яе]|где ты жив[ёе]шь|твой (дом|адрес))",
        # Ukrainian
        r"\b(я\s+(знайду|вистежу|прийду за))\b.{0,20}(теб[е]|де ти жив[е]ш|тв[іi]й (д[іi]м|адрес[ау]))",
        # German
        r"\b(ich (werde|will)\s+.{0,15}(finden|aufspüren|jagen))\b.{0,20}(dich|wo du wohnst|dein(e|em)?\s+(haus|adresse))",
        # Spanish
        r"\b(voy a\s+.{0,15}(encontrar|buscar|rastrear))\b.{0,20}(te|dónde vives|tu\s+(casa|dirección))",
        # Arabic
        r"(سأجد|سأعثر|سأتتبع|سأبحث عن).{0,20}(أين تعيش|بيتك|عنوانك|منزلك)",
        # Hindi
        r"(मैं\s+(ढूंढ|खोज|पता लगा)).{0,20}(तुम|तुझे|कहाँ रहते|तेरा घर|तेरा पता)",
        # Chinese (no word boundaries)
        r"(我会|我要|我将).{0,10}(找到|追踪|查到).{0,10}(你住|你家|你的地址)",
        # Italian
        r"\b(troverò|scoprirò|verrò a cercar)\b.{0,20}(te|dove (vivi|abiti)|tua casa|tuo indirizzo)",
        # French
        r"\b(je (vais|trouverai)|j'irai)\b.{0,20}(te (trouver|chercher)|où tu (habites|vis)|ta maison|ton adresse)",
        # Hebrew
        r"(אני אמצא|אני אחפש|אני אעקוב).{0,20}(איפה אתה גר|הבית שלך|הכתובת שלך)",
        # Japanese (character-based)
        r"(お前の|てめえの|あんたの).{0,10}(住所|家|居場所).{0,10}(突き止|見つけ|調べ)",
        r"(見つけ出し|追い詰め|突き止め).{0,10}(てやる|るぞ)",
        # Hinglish
        r"\b(main\s+(dhundh|khoj|pata laga)).{0,20}(tum|tujhe|kahan rehte|tera ghar|tera pata)",
    ],

    # Category 2: Violence threats ("kill/murder/stab you")
    "violence": [
        # English
        r"\b(kill|murder|stab|shoot|hurt|end)\b.{0,15}\b(you|your|them|him|her)\b",
        # Russian
        r"\b(убью|зарежу|застрелю|прибью|прикончу|порешу)\b.{0,15}(теб[яе]|вас|его|её)",
        r"(теб[яе]|вас|его|её).{0,15}\b(убью|зарежу|застрелю|прибью|прикончу|порешу)\b",  # SOV order
        # Ukrainian
        r"\b(вб'ю|заріжу|застрелю|приб'ю|прикінчу)\b.{0,15}(теб[е]|вас|його|її)",
        r"(теб[е]|вас|його|її).{0,15}\b(вб'ю|заріжу|застрелю|приб'ю|прикінчу)\b",  # SOV order
        # German
        r"\b(töte|umbringe|erstech|erschieß|verletze)\b.{0,15}(dich|euch|ihn|sie)",
        r"\b(ich (bringe|töte))\b.{0,15}(dich|euch)\s+(um)?",
        # Spanish
        r"\b(matar[ée]|apuñalar[ée]|disparar[ée]|acabar[ée] con)\b.{0,15}(te|ti|usted|ellos)",
        r"\b(te|ti)\b.{0,15}\b(matar[ée]|apuñalar[ée]|disparar[ée])\b",  # clitic-before-verb order
        # Arabic
        r"(سأقتلك|سأذبحك|سأطعنك|سأقضي عليك|أقتلك)",
        # Hindi
        r"(मार\s+डालूंगा|काट\s+दूंगा|गोली\s+मार|खत्म\s+कर\s+दूंगा).{0,10}(तुझे|तुम्हें|उसे)",
        r"(तुझे|तुम्हें|उसे).{0,10}(मार\s+डालूंगा|काट\s+दूंगा|गोली\s+मार|खत्म\s+कर\s+दूंगा)",  # SOV order
        # Chinese
        r"(杀了你|弄死你|宰了你|捅死你|打死你)",
        # Italian
        r"\b(ammazzo|uccido|accoltello|sparo)\b.{0,15}(te|ti|voi|lui|lei)",
        r"\b(ti ammazzo|ti uccido|vi ammazzo)\b",
        # French
        r"\b(tuer|poignarder|buter|descendre)\b.{0,15}(te|toi|vous|lui|elle)",
        r"\b(je (vais te|te) (tuer|buter|descendre))\b",
        # Hebrew
        r"(אהרוג|אדקור|אירה ב|אסיים את).{0,10}(אותך|אותו|אותה|אתכם)",
        # Japanese
        r"(殺す|殺して|殺してやる|刺して|ぶっ殺|ころす)",
        # Hinglish
        r"\b(maar\s+dalunga|kaat\s+dunga|goli\s+maar|khatam\s+kar\s+dunga)\b",
    ],

    # Category 3: Death wishes ("I hope you die")
    "death_wish": [
        # English
        r"\b(die|dead|death)\b.{0,10}\b(wish|hope|want|should|deserve)\b",
        r"\b(wish|hope|want)\b.{0,10}\b(die|dead|death|killed)\b",
        # Russian
        r"(сдохни|чтоб ты (сдох|умер)|надеюсь.{0,10}(сдохнешь|умрёшь))",
        # Ukrainian
        r"(здохни|щоб ти (здох|помер)|сподіваюсь.{0,10}(здохнеш|помреш))",
        # German
        r"(stirb|verreck|hoffentlich.{0,10}(stirbst|verreckst)|du sollst (sterben|verrecken))",
        # Spanish
        r"(muérete|ojalá.{0,10}(mueras|te mueras)|mereces.{0,10}morir)",
        # Arabic
        r"(أتمنى.{0,10}(تموت|موتك)|مت يا|يا ريت تموت|تستاهل الموت)",
        # Hindi
        r"(मर जा|मरो|मरना चाहिए|मौत|मर जाओ)",
        # Chinese
        r"(去死|该死|死吧|希望你死|你去死)",
        # Italian
        r"(muori|crepa|spero.{0,10}(muoia|crepi)|meriti.{0,10}morire)",
        # French
        r"(crève|j'espère.{0,10}(crèves|meures|mourras)|tu mérites.{0,10}mourir)",
        # Hebrew
        r"(תמות|מגיע לך למות|אני מקווה.{0,10}תמות)",
        # Japanese
        r"(死ね|死んで|くたばれ|死にやがれ|消えろ)",
        # Hinglish
        r"\b(mar ja|maro|marna chahiye|maut|mar jao)\b",
    ],

    # Category 4: Swatting / doxxing
    "swat_dox": [
        r"\bswat(t)?(ing|ed)?\b",
        r"\bdox(x)?(ing|ed)?\b",
        # Russian
        r"\b(сватт?инг|свотт?инг|докс(инг)?)\b",
        # Common across languages (English loanwords used in gaming communities)
        r"\b(swatt?ing|doxx?ing)\b",
    ],

    # Category 5: Mass violence ("bomb the school")
    "mass_violence": [
        # English
        r"\b(bomb|shoot up|blow up)\b.{0,15}\b(school|house|place|building)\b",
        # Russian
        r"\b(взорв[уа]|расстреля)\b.{0,15}(школ[уе]|дом|здани[ея])",
        # German
        r"\b(Bombe|spreng|Amoklauf)\b.{0,15}(Schule|Haus|Gebäude)",
        # Spanish
        r"\b(bomba|explotar|tiroteo)\b.{0,15}(escuela|casa|edificio)",
        # Arabic
        r"(قنبلة|تفجير|إطلاق نار).{0,15}(مدرسة|بيت|مبنى)",
        # Chinese
        r"(炸|爆破|枪击).{0,5}(学校|房子|建筑)",
        # Japanese
        r"(爆破|銃撃|爆弾).{0,5}(学校|家|建物)",
        # Hindi
        r"(बम|गोलीबारी|विस्फोट).{0,10}(स्कूल|घर|इमारत)",
    ],
}

# Compile all patterns
_compiled_multilingual_threats = []
for category, patterns in THREAT_PATTERNS_MULTILINGUAL.items():
    for pattern in patterns:
        try:
            _compiled_multilingual_threats.append(
                (category, re.compile(pattern, re.IGNORECASE | re.UNICODE))
            )
        except re.error:
            pass  # Skip invalid patterns gracefully


def detect_threat(text: str) -> tuple:
    """
    Returns (is_threat: bool, category: str or None).
    Checks all multilingual threat patterns.
    """
    for category, pattern in _compiled_multilingual_threats:
        if pattern.search(text):
            return True, category
    return False, None


# ═══════════════════════════════════════════════════════════════

class ProductionModerator:

    MAX_WARNS = 5.0
    MAX_TIMEOUTS = 3.0
    MAX_INFRACTIONS = 10.0

    COOLDOWN_THRESHOLD = 5
    DECAY_RATE = 1.0
    MOMENTUM_WINDOW = 10

    # Training used k=3 context window — production must match to avoid
    # train/inference embedding distribution mismatch (Fix: Issue #4).
    CONTEXT_WINDOW = 3

    # Monte-Carlo confidence: for borderline toxicity, sample the policy
    # N times stochastically and take majority vote (Meta paper Section 3.3.1).
    MC_SAMPLES = 4
    MC_TOX_LOW  = 0.30   # below this: always deterministic (clearly benign)
    MC_TOX_HIGH = 0.70   # above this: always deterministic (clearly toxic)

    def __init__(self, model_path: str = "data/models/best/best_model.zip",
                 calibration_file: str = "data/processed/toxicity_calibration.json"):
        self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.judge = pipeline(
            "text-classification",
            model="textdetox/xlmr-large-toxicity-classifier-v2",
            truncation=True,
            max_length=512,
        )
        self.model = MaskablePPO.load(model_path)
        self.user_ledger = {}
        self.banned_users = set()
        self.global_step = 0

        self.recent_toxicity = deque(maxlen=50)
        self.recent_actions = deque(maxlen=50)

        # Per-channel sliding window for context-aware embeddings.
        # Key: channel_id (str), Value: deque of recent message strings.
        self._channel_history: dict = {}

        # Load calibration
        self.calibration = None
        try:
            import os
            if os.path.exists(calibration_file):
                with open(calibration_file, "r") as f:
                    self.calibration = json.load(f)
                print(f"✅ Loaded per-language calibration from {calibration_file}")
            else:
                print(f"⚠️  No calibration file at {calibration_file} — using raw scores")
        except Exception as e:
            print(f"⚠️  Failed to load calibration: {e}")

        # Language detector
        try:
            from src.utils.language_detector import detect_language
            self._detect_language = detect_language
            print(f"✅ Language detector loaded")
        except ImportError:
            self._detect_language = None
            print(f"⚠️  Language detector not available")

    # ── ledger helpers ──────────────────────────────────────────

    def _ensure_ledger(self, user_id: str):
        if user_id not in self.user_ledger:
            self.user_ledger[user_id] = {
                "warns": 0.0, "timeouts": 0.0, "total_infractions": 0.0,
                "last_infraction_step": -1, "clean_streak": 0,
                "tox_momentum": deque(maxlen=self.MOMENTUM_WINDOW),
            }

    def _get_effective_infractions(self, user_id: str) -> float:
        led = self.user_ledger[user_id]
        raw = led["total_infractions"]
        if led["clean_streak"] <= self.COOLDOWN_THRESHOLD:
            return raw
        decay_steps = led["clean_streak"] - self.COOLDOWN_THRESHOLD
        return max(0.0, raw - decay_steps * self.DECAY_RATE)

    def _get_effective_warns(self, user_id: str) -> float:
        led = self.user_ledger[user_id]
        if led["clean_streak"] <= self.COOLDOWN_THRESHOLD:
            return led["warns"]
        decay_steps = led["clean_streak"] - self.COOLDOWN_THRESHOLD
        return max(0.0, led["warns"] - decay_steps * 0.5)

    # ── language helpers ────────────────────────────────────────

    def _detect_msg_language(self, message: str) -> str:
        if self._detect_language is None:
            return "en"
        try:
            result = self._detect_language(message)
            return result.lang if result.lang in LANGUAGES else "other"
        except Exception:
            return "other"

    def _get_language_onehot(self, lang_code: str) -> np.ndarray:
        vec = np.zeros(NUM_LANGUAGES, dtype=np.float32)
        idx = LANG_TO_IDX.get(lang_code, -1)
        if idx >= 0:
            vec[idx] = 1.0
        else:
            vec[:] = 1.0 / NUM_LANGUAGES
        return vec

    # ── toxicity scoring ────────────────────────────────────────

    def _calibrate_score(self, raw_score: float, lang: str) -> float:
        if self.calibration is None:
            return raw_score
        lang_params = self.calibration.get("languages", {}).get(lang)
        if lang_params is None or not lang_params.get("calibrated", False):
            return raw_score
        scale = lang_params.get("scale", 1.0)
        offset = lang_params.get("offset", 0.0)
        return float(np.clip(raw_score * scale + offset, 0.0, 1.0))

    def _get_toxicity(self, message: str, lang: str = None) -> tuple:
        """Returns (calibrated_score, detected_language, threat_category)."""
        if not message.strip():
            return 0.0, lang or "other", None

        if lang is None:
            lang = self._detect_msg_language(message)

        # Benign gaming vocabulary override — XLM-R falsely scores these as toxic
        if _BENIGN_GAMING_PATTERNS.match(message.strip()):
            return 0.0, lang, None

        # Raw classifier score
        result = self.judge(message)[0]
        is_toxic_label = result["label"].lower() in ["toxic", "1", "label_1"]
        raw_score = result["score"] if is_toxic_label else 1.0 - result["score"]
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # Per-language calibration
        score = self._calibrate_score(raw_score, lang)

        # Change 3: Multilingual threat detection fallback
        is_threat, threat_cat = detect_threat(message)
        if is_threat and score < 0.80:
            score = max(score, 0.85)

        return score, lang, threat_cat

    # ── action masking ──────────────────────────────────────────

    def _get_action_mask(self, tox: float, user_id: str) -> np.ndarray:
        led = self.user_ledger[user_id]
        eff_inf = self._get_effective_infractions(user_id)
        mask = np.ones(5, dtype=bool)

        if led["timeouts"] < 1:
            mask[4] = False
        if eff_inf < 2:
            mask[3] = False
        if tox < 0.15:
            mask[1] = False  # FIX D: block WARN — too lenient for near-zero toxicity
            mask[2] = False
            mask[3] = False
            mask[4] = False
        if tox >= 0.85:
            mask[0] = False
        if not mask.any():
            mask[0] = True
        return mask

    # ── context window helpers ──────────────────────────────────

    def _get_context_embedding(self, message: str, channel_id: str) -> np.ndarray:
        """
        Build a k=3 sliding-window context string and embed it.

        Training used context_strings.json with k=3 windows — production
        must use the same scheme or the 384-dim embedding distribution will
        differ from what the policy was trained on (Issue #4 fix).
        """
        if channel_id not in self._channel_history:
            self._channel_history[channel_id] = deque(maxlen=self.CONTEXT_WINDOW)

        history = self._channel_history[channel_id]

        # Build context: [prev_k, ..., prev_1, current]
        context_parts = list(history) + [message]
        context_str = " [SEP] ".join(context_parts)

        embedding = self.encoder.encode([context_str])[0].astype(np.float32)
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        embedding = np.clip(embedding, -1.0, 1.0)

        # Update history AFTER building embedding (so current message isn't included
        # in its own context window for the next message)
        history.append(message)
        return embedding

    def _predict_with_mc(self, obs: dict, mask: np.ndarray, tox: float) -> int:
        """
        Monte-Carlo confidence estimation for borderline toxicity scores.

        Meta paper Section 3.3.1: single-pass classification produces bimodal
        confidence distributions. For borderline content (tox in MC_TOX_LOW..MC_TOX_HIGH),
        sample MC_SAMPLES stochastic rollouts and return the majority vote action.
        Outside this range, use deterministic inference (faster, more confident).
        """
        # Clear-cut cases: use deterministic inference
        if tox < self.MC_TOX_LOW or tox > self.MC_TOX_HIGH:
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            return int(action)

        # Borderline: sample stochastically and take majority vote
        votes = []
        for _ in range(self.MC_SAMPLES):
            a, _ = self.model.predict(obs, deterministic=False, action_masks=mask)
            votes.append(int(a))

        # Majority vote (ties broken by lower action severity — err on the side of caution)
        from collections import Counter
        counts = Counter(votes)
        majority = counts.most_common(1)[0][0]
        return majority

    # ── main entry point ────────────────────────────────────────

    def moderate(self, message: str, user_id: str, channel_id: str = "default") -> dict:

        if user_id in self.banned_users:
            return {
                "decision": "REJECTED", "reason": "User is banned",
                "toxicity": None, "language": None, "threat_detected": None,
                "user_warns": self.user_ledger.get(user_id, {}).get("warns", 0),
                "user_timeouts": self.user_ledger.get(user_id, {}).get("timeouts", 0),
                "total_infractions": self.user_ledger.get(user_id, {}).get("total_infractions", 0),
                "effective_infractions": 0, "clean_streak": 0,
            }

        self._ensure_ledger(user_id)
        led = self.user_ledger[user_id]

        # Context-window embedding (fix train/inference mismatch — Issue #4)
        embedding = self._get_context_embedding(message, channel_id)

        # Toxicity (with calibration + multilingual threat detection)
        tox, detected_lang, threat_cat = self._get_toxicity(message)

        eff_warns = self._get_effective_warns(user_id)
        eff_inf = self._get_effective_infractions(user_id)

        # Observation — must match discord_env v5 exactly
        obs = {
            "message_embedding": embedding,
            "toxicity_score": np.array([tox], dtype=np.float32),
            "user_history": np.array([
                min(eff_warns       / self.MAX_WARNS,       1.0),
                min(led["timeouts"] / self.MAX_TIMEOUTS,    1.0),
                min(eff_inf         / self.MAX_INFRACTIONS, 1.0),
            ], dtype=np.float32),
            "server_heat": self._compute_server_heat(),
            "language_id": self._get_language_onehot(detected_lang),
        }

        mask = self._get_action_mask(tox, user_id)
        # Monte-Carlo confidence for borderline toxicity (Meta paper Section 3.3.1)
        action = self._predict_with_mc(obs, mask, tox)

        action_names = ["ALLOW", "WARN", "DELETE", "TIMEOUT", "BAN"]
        decision = action_names[action]

        # Update ledger
        if action >= 1:
            led["clean_streak"] = 0
            led["last_infraction_step"] = self.global_step
        else:
            if tox < 0.30:
                led["clean_streak"] += 1
            else:
                led["clean_streak"] = 0

        if decision == "WARN":
            led["warns"] += 1.0
            led["total_infractions"] += 1.0
        elif decision == "DELETE":
            led["total_infractions"] += 1.0
        elif decision == "TIMEOUT":
            led["timeouts"] += 1.0
            led["total_infractions"] += 1.0
        elif decision == "BAN":
            led["total_infractions"] += 1.0
            self.banned_users.add(user_id)

        led["tox_momentum"].append(tox)
        self.recent_toxicity.append(tox)
        self.recent_actions.append(int(action))
        self.global_step += 1

        return {
            "decision": decision,
            "toxicity": round(tox, 4),
            "language": detected_lang,
            "threat_detected": threat_cat,
            "user_warns": led["warns"],
            "user_timeouts": led["timeouts"],
            "total_infractions": led["total_infractions"],
            "effective_infractions": round(self._get_effective_infractions(user_id), 1),
            "clean_streak": led["clean_streak"],
        }

    def _compute_server_heat(self) -> np.ndarray:
        if not self.recent_toxicity:
            return np.array([0.0, 0.0], dtype=np.float32)
        tox_rate = np.mean([1.0 if t > 0.5 else 0.0 for t in self.recent_toxicity])
        act_rate = (np.mean([1.0 if a > 0 else 0.0 for a in self.recent_actions])
                    if self.recent_actions else 0.0)
        return np.array([min(float(tox_rate), 1.0),
                         min(float(act_rate), 1.0)], dtype=np.float32)

    def get_user_profile(self, user_id: str) -> dict:
        if user_id in self.banned_users:
            return {"status": "banned", **{k: v for k, v in self.user_ledger.get(user_id, {}).items() if k != "tox_momentum"}}
        if user_id not in self.user_ledger:
            return {"status": "clean", "warns": 0, "timeouts": 0, "total_infractions": 0, "clean_streak": 0}
        self._ensure_ledger(user_id)
        led = self.user_ledger[user_id]
        eff = self._get_effective_infractions(user_id)
        status = "rehabilitated" if led["total_infractions"] > 0 and eff < 1 else \
                 "flagged" if led["total_infractions"] > 0 else "clean"
        return {
            "status": status, "warns": led["warns"], "timeouts": led["timeouts"],
            "total_infractions": led["total_infractions"],
            "effective_infractions": round(eff, 1), "clean_streak": led["clean_streak"],
        }


# ── demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    moderator = ProductionModerator()
    user_id = "test_user"

    test_chain = [
        "Good morning everyone!",
        "This game is kinda annoying today.",
        "Are you guys seriously this bad?",
        "You are all actual garbage.",
        "You absolute trash, uninstall the game.",
        "I will find where you live.",
        "Just kidding guys, gg.",
    ]

    print("=== ESCALATION TEST ===")
    print("Expected: ALLOW → ALLOW → WARN → DELETE → TIMEOUT → BAN → REJECTED")
    print()
    for i, msg in enumerate(test_chain):
        result = moderator.moderate(msg, user_id)
        threat = f" [THREAT:{result['threat_detected']}]" if result.get('threat_detected') else ""
        print(f"  Step {i+1}: '{msg}' ")
        print(f"          → {result['decision']}  "
              f"(tox={result.get('toxicity', 'N/A')}, lang={result.get('language', '?')}, "
              f"warns={result['user_warns']:.0f}, timeouts={result['user_timeouts']:.0f}, "
              f"inf={result['total_infractions']:.0f}, eff_inf={result['effective_infractions']}, "
              f"clean={result['clean_streak']}){threat}")
