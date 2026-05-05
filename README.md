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

# Learning When to Warn, When to Ban
### Constrained Reinforcement Learning for Multilingual Content Moderation

**MURL 2026 — Maynooth University Reinforcement Learning Conference**  
Rishabh Jain · Department of Computer Science, Maynooth University

📄 [Paper](paper/main_murl.pdf) &nbsp;|&nbsp; 🖼️ [Poster](paper/poster.pdf) &nbsp;|&nbsp; 🤗 [Live Demo](https://huggingface.co/spaces/Rishabh12j/multilingual-rl-moderation)

---

## The Problem

Automated content moderation on multilingual platforms (e.g. Discord) is broken in three ways:

- **No user history.** Every message is scored in isolation — a user who sends ten borderline messages in a row gets the same response as a first-time offender.
- **Binary decisions only.** The output is always *remove* or *permit* — there is no mechanism to warn, temporarily suspend, or escalate gradually.
- **Threshold gaming.** Persistent harassers who keep each message just below the toxicity cutoff are never actioned, no matter how many times they repeat the pattern.

On top of this, multilingual toxicity classifiers (XLM-R) carry systematic per-language bias: Arabic is under-scored by −0.139 relative to English-equivalent content; Hinglish is over-scored by +0.095. Threshold-based systems bake this bias into every moderation decision.

---

## Our Approach

We frame moderation as a **Constrained Markov Decision Process (CMDP)** and train a single **MaskablePPO** policy that operates over a 5-level escalation ladder:

```
ALLOW → WARN → DELETE → TIMEOUT → BAN
```

Three design decisions distinguish this approach:

| Decision | What it does |
|---|---|
| **Hard action masking** | Illegal escalations (e.g. BAN before a prior TIMEOUT) are removed from the policy's action distribution *before* sampling — not penalised after the fact. |
| **Per-language affine calibration** | XLM-R toxicity scores are recalibrated per language using 75th-percentile anchors, neutralising classifier bias before it reaches the policy. |
| **Dual fairness protection** | An adaptive Lagrangian multiplier suppresses false positives; a Disparate Impact wrapper penalises any language whose ban rate exceeds 1.5× the baseline — both are training-time objectives, not post-hoc audits. |

---

## Results

| System | Accuracy | FN rate | TIMEOUT/BAN |
|---|---|---|---|
| Keyword filter | 0.673 | 0.272 | ✗ |
| Static threshold | 0.770 | 0.000 | ✗ |
| **RL agent (ours)** | **0.736** | **10⁻⁴** | **✓** |

The RL agent is the **only system tested that correctly issues TIMEOUT and BAN**, with per-class accuracies of 94.1% (DELETE), 91.6% (TIMEOUT), and 89.5% (BAN).  
Fairness audit: **12 of 14 language categories pass the 1.5× parity threshold**. The residual Hindi/Hinglish gap is traced to upstream XLM-R classifier bias, not policy bias.

---

## Replication

### 1. Environment setup

```bash
conda create -n moderation python=3.11 -y
conda activate moderation
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys (only needed for data generation and the Discord bot):

```bash
cp .env.example .env
```

### 2. Data pipeline (skip if using pre-built episodes)

The processed episodes are already in `data/processed/`. Re-run the pipeline only if you want to regenerate from scratch:

```bash
python -m src.pipeline.data_ingestion
python -m src.pipeline.semantic_clustering
python -m src.pipeline.causal_routing_baseten   # requires Baseten API key
python -m src.pipeline.user_ledger
python -m src.pipeline.sliding_window
python -m src.pipeline.add_language_field
python -m src.pipeline.precompute_embeddings
python -m src.pipeline.precompute_toxicity
python -m src.pipeline.episode_mapping
python -m src.pipeline.split_episodes
```

### 3. Train

```bash
# Stage 1: behaviour cloning from oracle escalation rule
python -m src.agent.behavior_clone

# Stage 2: MaskablePPO fine-tune with Lagrangian + Disparate Impact wrappers
python -m src.agent.train
```

The best checkpoint is saved to `data/models/best/best_model.zip`.

### 4. Evaluate

```bash
# Held-out test split accuracy + per-class breakdown
python -m src.agent.evaluate data/models/best/best_model.zip

# Baseline comparison (keyword filter, static threshold vs RL agent)
python -m src.diagnostics.baseline_comparison

# Cross-lingual fairness audit (per-language BAN rates)
python -m src.diagnostics.crosslingual_parity

# Trajectory verification (escalation + sustained-troll scenarios)
python -m src.diagnostics.procedural_scenarios
```

### 5. Discord bot (optional)

Requires `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, and `DISCORD_TEST_CHANNEL_ID` in `.env`:

```bash
python -m src.agent.discord_bot
```

### 6. Local demo (Gradio)

```bash
python app.py
```

---

## Repository Layout

```
src/
  pipeline/        Data ETL — ingestion, clustering, LLM synthesis, embedding, splitting
  env/             Gymnasium environment, Lagrangian + Disparate Impact wrappers, action masking
  agent/           behavior_clone.py, train.py, evaluate.py, production_moderator.py, discord_bot.py
  diagnostics/     baseline_comparison.py, crosslingual_parity.py, procedural_scenarios.py
  utils/           Language detection helpers

data/
  processed/       Pre-built episodes (595 episodes / 8,016 steps), calibration JSON
  models/best/     Trained policy checkpoint (best_model.zip)

paper/
  main_murl.tex    MURL 2026 camera-ready source
  main_murl.pdf    Compiled paper
  poster.tex       A0 conference poster source
  poster.pdf       Compiled poster

app.py             Gradio demo
requirements.txt   Python dependencies
Dockerfile         Container for HuggingFace Spaces deployment
```

---

## Citation

```bibtex
@inproceedings{jain2026fairmod,
  title     = {Learning When to Warn, When to Ban: Constrained Reinforcement Learning
               for Multilingual Content Moderation},
  author    = {Jain, Rishabh},
  booktitle = {2nd Maynooth University Reinforcement Learning Conference (MURL)},
  year      = {2026}
}
```
