FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models at build time (cached in image)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
from transformers import pipeline; \
pipeline('text-classification', model='textdetox/xlmr-large-toxicity-classifier-v2')"

# Copy source and required data
COPY src/ src/
COPY data/models/best/ data/models/best/
COPY data/processed/toxicity_calibration.json data/processed/toxicity_calibration.json
COPY .env.example .env.example

# Create data directory for SQLite ledger
RUN mkdir -p data/bot

# Expose no ports (bot connects outbound to Discord gateway)

# Default: interactive production demo
# Override to run the Discord bot:
#   docker run -d --env-file .env fairmod python -m src.bot.discord_bot
ENTRYPOINT ["python", "-m", "src.agent.production_moderator"]
