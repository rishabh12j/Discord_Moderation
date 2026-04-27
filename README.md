---
title: Multilingual Rl Moderation
emoji: 🐨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# FairMod: Multilingual RL Moderation Agent

Production-grade Discord moderation system using Constrained Markov Decision Processes (CMDP) with MaskablePPO, Lagrangian fairness constraints, and multilingual threat detection across 13 languages. Deployed as a live Discord bot with persistent state and a full audit trail.

## Architecture

The agent processes Discord messages through a three-stage pipeline:

**Stage 1 -- Feature Extraction.** Each message is encoded by `paraphrase-multilingual-MiniLM-L12-v2` (384-dim embedding) and scored by `textdetox/xlmr-large-toxicity-classifier-v2` (toxicity probability). Per-language affine calibration normalizes classifier confidence across languages. A context window (k=3) of recent channel messages is concatenated before embedding to match training conditions.

**Stage 2 -- State Construction.** The observation combines the embedding, calibrated toxicity score, per-user infraction history (warns, timeouts, effective infractions with cool-down decay), server-wide toxicity rate, and a 13-dim language one-hot vector. Total observation: 403 dimensions.

**Stage 3 -- Policy Decision.** MaskablePPO selects from 5 actions (ALLOW, WARN, DELETE, TIMEOUT, BAN) subject to dynamic action masks that enforce graduated escalation. The agent cannot BAN without prior TIMEOUT, cannot TIMEOUT without 2+ prior infractions, cannot ALLOW clearly toxic content (score >= 0.85), and cannot WARN/DELETE/TIMEOUT/BAN on clearly benign content (score < 0.15). For borderline toxicity (0.30-0.70), Monte-Carlo confidence estimation via 4-sample stochastic voting replaces deterministic inference.

## CMDP Formulation

The base MDP is extended with Lagrangian constraints:

**Reward:** `R_t = R_base + R_escalation - lambda * C_t` where `C_t` is a false-positive cost and `R_escalation` is a rubric-based trajectory quality component.

**Lagrangian multiplier lambda** is updated every 500 steps: increases if FP rate exceeds 5%, slowly decays otherwise.

**Disparate Impact Penalty:** If `P(BAN|lang=X) > 1.5 * P(BAN)`, a penalty of -2.0 is applied, preventing the agent from over-moderating any single language.

## Two-Stage Training (BC -> RL)

Following findings from [Liu et al. (2025)](https://arxiv.org/abs/2512.20061) on scaling RL for content moderation:

**Stage 1 -- Behavior Cloning:** An oracle escalation rule is rolled out across all training episodes. The MaskablePPO policy is trained via NLL loss on (observation, oracle_action) pairs for 30 epochs, anchoring the policy in the correct behavioral structure.

**Stage 2 -- RL Fine-Tuning:** The BC-initialized policy is loaded and fine-tuned via MaskablePPO for 1M timesteps with n_steps=4096, batch_size=256, giving an effective batch of 16,384 per update. The RL stage can improve *beyond* the oracle rule.

Constraint-aware checkpoint selection saves the best model based on `reward - 10 * max(0, FP_rate - 0.05)`, not reward alone.

## Graduated Escalation

The agent enforces a strict ladder: ALLOW -> WARN -> DELETE -> TIMEOUT -> BAN.

Action masks structurally prevent skipping tiers. Cool-down decay allows rehabilitation: after 5 consecutive clean messages, effective infractions decrease by 1 per additional clean message, eventually returning a user to clean status.

## Multilingual Support

**13 languages:** English, Russian, Ukrainian, German, Spanish, Arabic, Hindi, Chinese, Italian, French, Hebrew, Japanese, Hinglish.

**Per-language toxicity calibration** normalizes XLM-R confidence distributions so that the 85th percentile toxic score in Arabic maps to the same value as in English.

**Multilingual threat detection** covers 5 categories (stalking, violence, death wishes, swatting/doxxing, mass violence) with regex patterns in all 13 languages, including SOV word-order variants for Russian, Ukrainian, Spanish, and Hindi.

## Project Structure

```
src/
  pipeline/                    # Phase 1-2: Data ETL
    data_ingestion.py            # Multilingual dataset loading (textdetox)
    semantic_clustering.py       # K-Means on MiniLM embeddings
    causal_routing_baseten.py    # LLM thread generation via DeepSeek
    user_ledger.py               # Per-user baseline ledger
    etl_validator.py             # Data quality checks
    sliding_window.py            # k=3 context window construction
    add_language_field.py        # Language tagging from thread IDs
    precompute_embeddings.py     # 384-dim vectors -> context_embeddings.npy
    precompute_toxicity.py       # Calibrated scores -> toxicity_scores.npy
    episode_mapping.py           # Thread to episode state mapping
    split_episodes.py            # 80/20 train/test split
    validate_pipeline.py         # End-to-end pipeline validation
  env/                         # Phase 3: CMDP Environment
    discord_env.py               # Gymnasium env (403-dim obs, 5 actions, masking)
    wrappers.py                  # Lagrangian, DisparateImpact, RewardScaling, Metrics
  agent/                       # Phase 4: Training and Inference
    behavior_clone.py            # Stage 1: BC pre-training (oracle imitation)
    train.py                     # Stage 2: MaskablePPO RL fine-tuning
    evaluate.py                  # Deterministic evaluation
    production_moderator.py      # Real-time inference with MC confidence
  bot/                         # Phase 5: Discord Bot
    discord_bot.py               # discord.py bot with enforcement actions
    user_ledger_db.py            # SQLite persistent user state + audit log
  diagnostics/                 # Phase 6: Evaluation
    crosslingual_parity.py       # 10-sentence x 13-language parity test
    procedural_scenarios.py      # Escalation, troll, rehabilitation tests
    baseline_comparison.py       # Keyword + threshold baselines + fairness
    vectorized_eval.py           # Parallel episode evaluation
  utils/
    language_detector.py         # langdetect + Lingua + INLD
    toxicity_classifier.py
  pipeline_ci.py               # CI/CD orchestrator
```

## Quick Start

### 1. Install

```bash
conda create -n moderation python=3.11 -y
conda activate moderation
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
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
```

### 3. Split Episodes (Train/Test)

```bash
python -m src.pipeline.split_episodes
```

Creates `episodes_train.json` (80%) and `episodes_test.json` (20%).

### 4. Train (Two-Stage)

```bash
# Stage 1: Behavior Cloning (~5 min) — creates data/models/bc_init.zip
python -m src.agent.behavior_clone

# Stage 2: RL Fine-Tuning (~30 min) — auto-loads bc_init.zip if present
python -m src.agent.train
```

### 5. Evaluate

```bash
python -m src.agent.evaluate data/models/best/best_model.zip
python -m src.diagnostics.crosslingual_parity
python -m src.diagnostics.procedural_scenarios
python -m src.diagnostics.baseline_comparison
python -m src.diagnostics.vectorized_eval
```

### 6. Run Discord Bot

```bash
# Set environment variables
cp .env.example .env
# Edit .env with your Discord bot token and guild ID

export DISCORD_BOT_TOKEN=your_token_here
export DISCORD_GUILD_ID=your_guild_id_here
python -m src.bot.discord_bot
```

**Discord Developer Portal setup:**
- Create a bot at https://discord.com/developers/applications
- Enable **Message Content Intent** (Privileged Gateway Intents)
- Enable **Server Members Intent**
- Invite with permissions: Read Messages, Send Messages, Manage Messages, Moderate Members, Ban Members

**Bot commands** (require Manage Messages / Manage Guild):
| Command | Description |
|---------|-------------|
| `!modlogs [N]` | Last N moderation actions in the guild |
| `!modprofile @user` | User's infraction profile and history |
| `!modreset @user` | Reset a user's ledger (Manage Guild) |
| `!modstatus` | Bot health, tracked users, action distribution |

### 7. Production Demo (standalone, no Discord)

```bash
python -m src.agent.production_moderator
```

### 8. CI/CD Pipeline (full automated run)

```bash
python -m src.pipeline_ci
```

### 9. Docker

```bash
docker build -t fairmod .

# Run as production demo
docker run -it fairmod

# Run as Discord bot
docker run -d --env-file .env fairmod python -m src.bot.discord_bot
```

## Evaluation Results

### Baseline Comparison

| System | Accuracy | FP Rate | FN Rate | TIMEOUT/BAN |
|--------|----------|---------|---------|-------------|
| Keyword Filter | 67.3% | 0.020 | 0.272 | No |
| Static Threshold | 77.0% | 0.000 | 0.000 | No |
| **FairMod (RL)** | **73.6%** | 0.153 | **<0.001** | **Yes** |

The RL agent is the only system capable of graduated enforcement (TIMEOUT/BAN). Per-class accuracy on enforcement actions: DELETE 94.1%, TIMEOUT 91.6%, BAN 89.5%. The lower overall accuracy reflects conservative over-warning at the benign boundary.

### Procedural Scenarios

**Escalation Chain:** ALLOW, ALLOW, ALLOW, WARN, DELETE, TIMEOUT, TIMEOUT, TIMEOUT, BAN, REJECTED

**Sustained Troll:** DELETE, ALLOW, DELETE, TIMEOUT, TIMEOUT, TIMEOUT via recidivism detection

**Rehabilitation:** DELETE, ALLOW (x11), effective infractions decay to 0, mild relapse treated leniently

### Multilingual Threat Detection

13/13 languages covered for all 5 threat categories (stalking, violence, death wishes, swatting/doxxing, mass violence). SOV word-order variants included for Russian, Ukrainian, Spanish, and Hindi. Zero false positives on benign gaming vocabulary.

### Cross-Lingual Ban Rate Parity

All 13 language groups pass the 1.5x ban-rate parity threshold. Hinglish reduced from 2.60x to 1.06x through calibration correction and DisparateImpactWrapper activation.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage BC -> RL | RL-Only without behavioral prior fails to converge (Liu et al. 2025) |
| Wrapper order: Lagrangian -> DI -> Metrics -> Scaling | Penalties computed in raw reward space; scaling is last to prevent -4.0 total |
| MC confidence (4 samples) | Single-pass inference overcommits on borderline content |
| Constraint-aware checkpoint | Prevents selecting high-reward models that violate FP constraint |
| WARN masked at tox < 0.15 | Eliminates false-positive warnings on clearly benign content |
| n_steps=4096, batch=256 | Effective batch 16,384 exceeds the ~1,024 inflection point for stable PPO |

## Known Limitations

1. **Mild toxicity blind spots.** XLM-R scores mild frustration near zero in Italian, Ukrainian, French, Russian, German, Arabic, and Hinglish. This is a classifier limitation, not fixable by calibration.

2. **No code-switching handling.** Messages mixing languages mid-sentence may confuse both the language detector and classifier.

3. **Training corpus size.** 595 episodes (8,016 steps) is relatively small. Disagreement-based reweighting would improve data efficiency.

4. **Hindi/Hinglish classifier bias.** XLM-R over-scores Hindi by +0.095 vs English. This compounds through the escalation ladder, producing BAN rate disparity (Hindi 1.92x, Hinglish 1.64x) that calibration cannot fully correct. Requires classifier replacement or fine-tuning.

5. **Conservative over-warning.** ~22% of benign messages receive WARN instead of ALLOW. This is a safety-conservative tradeoff — false WARNs are less harmful than false BANs.

## References

- Altman, E. (1999). *Constrained Markov Decision Processes*. CRC Press.
- Liu, Z. et al. (2025). Scaling RL for Content Moderation with LLMs. arXiv:2512.20061.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Huang, S. & Ontanon, S. (2022). A Closer Look at Invalid Action Masking. FLAIRS.
- Conneau, A. et al. (2020). Unsupervised Cross-Lingual Representation Learning at Scale. ACL.
- Dementieva, D. et al. (2023). Multilingual Text Detoxification Shared Task. ACL Workshop.
