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

# FairMod

MURL2026 project. RL-based Discord content moderation agent.

The agent picks one of `ALLOW / WARN / DELETE / TIMEOUT / BAN` for each message,
across 13 languages, using a MaskablePPO policy trained as a Constrained MDP
with Lagrangian fairness constraints.

Paper: [paper/main_murl.tex](paper/main_murl.tex)
Live demo: https://huggingface.co/spaces/Rishabh12j/multilingual-rl-moderation

## Run

```bash
conda create -n moderation python=3.11 -y
conda activate moderation
pip install -r requirements.txt
```

Pipeline (only needed if regenerating data):

```bash
python -m src.pipeline.data_ingestion
python -m src.pipeline.semantic_clustering
python -m src.pipeline.causal_routing_baseten
python -m src.pipeline.user_ledger
python -m src.pipeline.sliding_window
python -m src.pipeline.add_language_field
python -m src.pipeline.precompute_embeddings
python -m src.pipeline.precompute_toxicity
python -m src.pipeline.episode_mapping
python -m src.pipeline.split_episodes
```

Train:

```bash
python -m src.agent.behavior_clone   # BC pretraining
python -m src.agent.train            # MaskablePPO fine-tune
```

Evaluate:

```bash
python -m src.agent.evaluate data/models/best/best_model.zip
python -m src.diagnostics.baseline_comparison
python -m src.diagnostics.crosslingual_parity
python -m src.diagnostics.procedural_scenarios
```

Discord bot (needs `.env` with `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`,
`DISCORD_TEST_CHANNEL_ID`):

```bash
python -m src.agent.discord_bot
```

Local demo:

```bash
python app.py
```

## Layout

```
src/pipeline/        data ETL
src/env/             gym env + wrappers
src/agent/           behavior_clone, train, evaluate, production_moderator, discord_bot
src/diagnostics/     baseline + parity + scenario tests
src/utils/           language detection
data/models/best/    trained policy
data/processed/      calibration JSON
paper/               main_murl.tex (MURL2026 submission), main.tex (IEEE backup)
```
