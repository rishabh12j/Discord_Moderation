# Multilingual Discord Moderator RL

An automated, multilingual Discord moderation agent trained using Reinforcement Learning (RL) via a Constrained Markov Decision Process (CMDP). The agent learns to balance platform safety (minimizing toxicity) with user engagement, strictly avoiding language-based biases.

## Architecture Stack
* **Language:** Python 3.11.14
* **RL Framework:** `sb3-contrib` (MaskablePPO) + `gymnasium`
* **Embeddings:** `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Toxicity Judge (Reward Model):** `transformers` (`textdetox/xlmr-large-toxicity-classifier-v2`)
* **Language Detection:** `langdetect` + `lingua` (with Hinglish support via INLD)
* **Tracking:** Weights & Biases (`wandb`)

## Core Design Philosophy: Strict Language-Agnostic Policy
A major challenge with multilingual toxicity classifiers is the performance discrepancy (F1 score) across different languages. To prevent the RL agent from internalizing biases against low-resource or morphologically complex languages:
1. **Observation Space:** The policy network observes strictly content-driven features (message embeddings, global user history norms, and server stats). Language metadata and language-based user priors are intentionally excluded from the state.
2. **Fairness Auditing:** Language detection is performed under the hood to track the agent's actions (e.g., measuring disparate impact on ban rates).
3. **Uncertainty Guardrails:** The environment uses action-masking to prevent irreversible actions (like Banning) on unsupported languages when the toxicity score is highly ambiguous.

## Environment Dynamics (Gymnasium)
The environment steps through static datasets restructured into chronological "threads" (episode chunks). 
* **Action Space:** Discrete(4) -> `[0: ALLOW, 1: WARN, 2: DELETE, 3: BAN]`
* **User Simulator:** The environment alters future steps dynamically. Issuing a `WARN` reduces a user's future toxicity in the thread, while a `BAN` scrubs all their future messages (`[BANNED]`).

## Setup & Preprocessing (Phase 1 & 2)
1. `data_ingestion.py`: Pulls the TextDetox dataset and synthesizes threaded conversations.
2. `src/utils/build_user_norms.py`: Computes language-agnostic historical toxicity averages per user.
3. `src/utils/precompute_embeddings.py` & `precompute_scores.py`: Pre-processes all text through the MiniLM and XLM-R models to ensure lightning-fast RL training loops.

## Training & Rewards (Phase 3 & 4)
*TODO: Document Base Reward, Lagrangian constraints, and Fairnes/Disparate Impact penalties.*

## Evaluation (Phase 5)
*TODO: Document True Positive Rate vs False Positive Rate, and baseline comparisons (Keyword Filter vs. Threshold Classifier vs. RL Agent).*
