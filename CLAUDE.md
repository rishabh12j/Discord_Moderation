# Project: Multilingual RL Moderation Agent

CMDP-based Discord moderation system using MaskablePPO with graduated escalation (ALLOW→WARN→DELETE→TIMEOUT→BAN), Lagrangian fairness constraints, and multilingual support across 13 languages.

## Build and Run

```bash
conda activate moderation

# Data pipeline (run in order)
python -m src.pipeline.data_ingestion
python -m src.pipeline.semantic_clustering
python -m src.pipeline.causal_routing_baseten
python -m src.pipeline.user_ledger
python -m src.pipeline.etl_validator
python -m src.pipeline.sliding_window
python -m src.pipeline.add_language_field
python -m src.pipeline.precompute_embeddings
python -m src.pipeline.precompute_toxicity
python -m src.pipeline.episode_mapping
python -m src.pipeline.validate_pipeline

# Train
python -m src.agent.train

# Diagnostics
python -m src.diagnostics.crosslingual_parity
python -m src.diagnostics.procedural_scenarios
python -m src.diagnostics.baseline_comparison
python -m src.diagnostics.vectorized_eval

# Production demo
python -m src.agent.production_moderator

# Full CI/CD
python -m src.pipeline_ci
```

## Architecture

```
src/
  pipeline/          → Phase 1-2: ETL. Static data → .npy arrays + .json episodes
  env/               → Phase 3: Gymnasium CMDP environment (discord_env.py, wrappers.py)
  agent/             → Phase 4: MaskablePPO training + production inference
  diagnostics/       → Phase 5: All evaluation and red teaming
  utils/             → Language detection, toxicity classifier helpers
  pipeline_ci.py     → Phase 6: CI/CD orchestrator
```

### Data Flow
Raw text → `data_ingestion.py` → `semantic_clustering.py` → `causal_routing_baseten.py` (DeepSeek V3) → `sliding_window.py` → `add_language_field.py` → `precompute_embeddings.py` (384-dim MiniLM) → `precompute_toxicity.py` (XLM-R with per-language calibration) → `episode_mapping.py` → static .npy/.json consumed by `discord_env.py`

### Observation Space (403 dims)
- `message_embedding`: Box(384,) — paraphrase-multilingual-MiniLM-L12-v2
- `toxicity_score`: Box(1,) — calibrated XLM-R score
- `user_history`: Box(3,) — effective warns, timeouts, effective infractions (with cool-down decay)
- `server_heat`: Box(2,) — rolling toxicity rate, action rate
- `language_id`: Box(13,) — one-hot for en/ru/uk/de/es/ar/hi/zh/it/fr/he/ja/hin

### Action Space: Discrete(5)
ALLOW(0), WARN(1), DELETE(2), TIMEOUT(3), BAN(4)

### Action Masks
- BAN blocked until `timeouts >= 1`
- TIMEOUT blocked until `effective_infractions >= 2`
- DELETE/TIMEOUT/BAN blocked when `toxicity < 0.15`
- ALLOW blocked when `toxicity >= 0.85`

## Critical Invariants

- `discord_env.py` and `production_moderator.py` MUST mirror each other: observation construction, action masking, ledger logic, cool-down config. Any change to one must be applied to the other.
- Cool-down: `COOLDOWN_THRESHOLD=5`, `DECAY_RATE=1.0`. After 5 clean messages, effective infractions decay by 1 per clean message.
- Toxicity classifier: `textdetox/xlmr-large-toxicity-classifier-v2`. Score isolated `raw_text` only — never concatenated context strings.
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`. Do not substitute without retraining.
- Calibration params in `data/processed/toxicity_calibration.json` must be applied identically in training (`precompute_toxicity.py`) and inference (`production_moderator.py`).

## When Making Changes

- Observation space → edit BOTH `discord_env.py` AND `production_moderator.py`, then retrain
- Reward function → edit `discord_env.py` step(), retrain
- Action masks → edit BOTH `discord_env.py` action_masks() AND `production_moderator.py` _get_action_mask()
- Threat patterns → edit `production_moderator.py` THREAT_PATTERNS_MULTILINGUAL only (no retrain)
- Calibration → edit `precompute_toxicity.py`, regenerate toxicity_scores.npy, retrain

## Known Limitations

1. XLM-R scores mild toxicity near zero in Italian, Ukrainian, French, Russian, German, Arabic, Hinglish — classifier blind spot, not fixable by calibration.
2. Death threat regex incomplete for Russian, Ukrainian, Spanish, Hindi (4/13 missed).
3. No code-switching handling (mixing languages mid-sentence).

## 13 Languages
en, ru, uk, de, es, ar, hi, zh, it, fr, he, ja, hin — indexed in `LANG_TO_IDX` dict in both discord_env.py and production_moderator.py.