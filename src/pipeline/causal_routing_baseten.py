import os
import json
import argparse
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import random
from collections import Counter

# ── Baseten client — Model APIs endpoint ────────────────────
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://inference.baseten.co/v1",  # ✅ FIXED
)
MODEL = "deepseek-ai/DeepSeek-V3.1"  # exact Baseten model slug

# ── System prompts ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a synthetic dataset generator for an AI content moderation system.
Generate realistic Discord conversation threads for training purposes.

STRICT RULES:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Each thread must have exactly 8 messages from 3-4 different users.
3. THREAD TYPE determines content:
   - "toxic": thread MUST contain 3-5 genuinely toxic messages (harassment, slurs, threats, hate speech)
   - "escalating": starts benign, ends with 3+ toxic messages
   - "benign": normal conversation, zero toxicity
   - "subtle": borderline — sarcasm, passive aggression, dog whistles
   - "mild": mild frustration or light complaints — annoying but NOT severe (no slurs, no threats)
4. Preserve language — if seed texts are Russian, respond in Russian. Arabic → Arabic. etc.
5. Do NOT sanitize toxic content — the moderation model needs real examples.

Output format (strict):
{
  "thread_id": "unique-id-here",
  "thread_type": "toxic",
  "language": "en",
  "messages": [
    {"user_id": "UserA", "text": "message text here", "is_toxic": false},
    {"user_id": "UserB", "text": "message text here", "is_toxic": true}
  ]
}"""

# Specialized prompt for Hinglish threads — enforces Hindi-English code-switching
SYSTEM_PROMPT_HINGLISH = """You are a synthetic dataset generator for an AI content moderation system.
Generate realistic Discord conversation threads in Hinglish (Hindi-English code-switching).

STRICT RULES:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Each thread must have exactly 8 messages from 3-4 different users.
3. CRITICAL: ALL messages must use natural Hindi-English code-switching — mix Hindi words/phrases
   with English words in the same sentence. Do NOT write pure Hindi or pure English.
   Examples of natural Hinglish:
   - Benign: "yaar gg that was amazing", "bhai achha khela tune", "kya mast game tha yaar",
             "nahi yaar I don't know how to win this", "bhai theek hai next game mein dekhenge"
   - Mild frustration: "yaar kya kar rahe ho seriously", "bhai theek se khelo na",
                       "itna bura kyu khel rahe ho yaar", "arre come on yaar focus karo"
   - Toxic: actual Hinglish insults using code-switching
4. THREAD TYPE determines content:
   - "toxic": 3-5 genuinely toxic Hinglish messages
   - "escalating": starts benign Hinglish, ends with 3+ toxic Hinglish messages
   - "mild": mild frustration in Hinglish — annoying but NOT severe (no slurs, no threats)
   - "subtle": borderline sarcasm, passive aggression in Hinglish
   - "benign": normal Hinglish gaming conversation, zero toxicity
5. Do NOT sanitize toxic content — the moderation model needs real examples.

Output format (strict):
{
  "thread_id": "unique-id-here",
  "thread_type": "benign",
  "language": "hin",
  "messages": [
    {"user_id": "UserA", "text": "yaar gg that round was fun", "is_toxic": false},
    {"user_id": "UserB", "text": "haan bhai next game mein aur achha khelenge", "is_toxic": false}
  ]
}"""


def get_seed_texts(df: pd.DataFrame, lang: str, toxic: bool, n: int = 5) -> list:
    """Pull real examples from your clustered dataset as seeds."""
    subset = df[(df["language"] == lang) & (df["toxic"] == int(toxic))]
    if len(subset) == 0:
        subset = df[df["language"] == lang]
    if len(subset) == 0:
        return ["example message"]
    return subset["text"].sample(
        min(n, len(subset)), random_state=random.randint(0, 9999)
    ).tolist()


def generate_thread(
    seed_texts: list,
    thread_type: str,
    lang: str,
    retries: int = 2,
    system_prompt: str = None,
) -> dict | None:
    """Call DeepSeek V3 0324 via Baseten to generate one thread."""
    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    user_prompt = json.dumps({
        "instruction": f"Generate a {thread_type} Discord thread in language '{lang}'.",
        "seed_examples": seed_texts,
        "thread_type": thread_type,
        "language": lang,
    }, ensure_ascii=False)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=1200,
            )

            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences just in case
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            thread = json.loads(raw)

            # ── Validate structure ────────────────────────────
            assert "messages" in thread, "Missing messages"
            assert isinstance(thread["messages"], list), "messages must be a list"
            assert len(thread["messages"]) >= 4, f"Too few messages: {len(thread['messages'])}"
            for msg in thread["messages"]:
                assert "user_id" in msg and "text" in msg, f"Bad msg: {msg}"
                assert isinstance(msg["text"], str) and len(msg["text"].strip()) > 2

            # ── Enforce toxic presence for toxic/escalating ───
            if thread_type in ("toxic", "escalating"):
                toxic_count = sum(1 for m in thread["messages"] if m.get("is_toxic", False))
                if toxic_count == 0:
                    if attempt < retries - 1:
                        continue  # Retry — model sanitized it
                    return None

            thread["thread_type"] = thread_type
            thread["language"] = lang
            return thread

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
        except (AssertionError, KeyError, AttributeError) as e:
            return None

    return None


def generate_causal_threads(
    input_file: str = "data/processed/clustered_multilingual.csv",
    output_file: str = "data/processed/causal_threads.json",
    target_threads: int = 500,
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file} — run clustering first.")

    df = pd.read_csv(input_file)
    languages = df["language"].unique().tolist()
    print(f"📂 Loaded {len(df)} rows across languages: {languages}")

    # ── 50% toxic signal, 50% clean ──────────────────────────
    type_distribution = [
        ("toxic",      0.30),
        ("escalating", 0.20),
        ("subtle",     0.15),
        ("benign",     0.35),
    ]

    # Build generation plan
    plan = []
    for thread_type, ratio in type_distribution:
        count = int(target_threads * ratio)
        for _ in range(count):
            lang = random.choice(languages)
            is_toxic_seed = thread_type in ("toxic", "escalating")
            plan.append((thread_type, lang, is_toxic_seed))
    random.shuffle(plan)

    threads = []
    failed = 0

    print(f"\n🚀 Generating {len(plan)} threads using {MODEL}...")
    print(f"   Distribution: toxic=30%, escalating=20%, subtle=15%, benign=35%\n")

    for thread_type, lang, is_toxic_seed in tqdm(plan):
        seed_texts = get_seed_texts(df, lang, toxic=is_toxic_seed, n=5)
        thread = generate_thread(seed_texts, thread_type, lang)

        if thread:
            threads.append(thread)
        else:
            failed += 1

        # Be gentle on the API — small delay every 50 requests
        if (len(threads) + failed) % 50 == 0:
            time.sleep(0.5)

    # ── Stats ─────────────────────────────────────────────────
    print(f"\n✅ Generated {len(threads)} valid threads ({failed} failed/rejected)")
    counts = Counter(t.get("thread_type", "unknown") for t in threads)
    print("\n📊 Thread type breakdown:")
    for t_type, count in sorted(counts.items()):
        print(f"   {t_type:15s}: {count:4d} ({count/max(len(threads),1)*100:.1f}%)")

    lang_counts = Counter(t.get("language", "?") for t in threads)
    print("\n🌍 Language breakdown:")
    for lang, count in sorted(lang_counts.items()):
        print(f"   {lang:8s}: {count:4d}")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(threads)} threads → {output_file}")


def generate_supplement_threads(
    input_file: str = "data/processed/clustered_multilingual.csv",
    output_file: str = "data/processed/causal_threads.json",
    hinglish_threads: int = 40,
    mid_tox_threads_per_lang: int = 15,
):
    """
    Append targeted supplement threads to an existing causal_threads.json.

    Two passes:
    1. Hinglish pass — benign-heavy with code-switching prompt to fix 2.6× BAN disparity.
    2. Mid-toxicity pass — "mild" + "escalating" across all 13 languages to fill the
       0.15–0.85 toxicity gap (currently only 5.9% of training steps).

    Does NOT overwrite existing threads — appends and saves back.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file} — run semantic_clustering first.")

    df = pd.read_csv(input_file)

    # Load existing threads to append to
    existing_threads = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_threads = json.load(f)
    existing_ids = {t.get("thread_id", "") for t in existing_threads}
    print(f"📂 Loaded {len(existing_threads)} existing threads.")

    all_languages = ["en", "ru", "uk", "de", "es", "ar", "hi", "zh", "it", "fr", "he", "ja", "hin"]
    new_threads = []
    failed = 0

    def _unique_id(lang: str, thread_type: str, counter: list) -> str:
        counter[0] += 1
        candidate = f"{lang}-{thread_type}-supplement-{counter[0]}"
        while candidate in existing_ids or candidate in {t.get("thread_id") for t in new_threads}:
            counter[0] += 1
            candidate = f"{lang}-{thread_type}-supplement-{counter[0]}"
        return candidate

    counter = [0]

    # ── Pass 1: Hinglish — benign-heavy distribution ─────────
    hin_distribution = [
        ("benign",     0.20),
        ("mild",       0.30),
        ("subtle",     0.20),
        ("escalating", 0.20),
        ("toxic",      0.10),
    ]
    hin_plan = []
    for thread_type, ratio in hin_distribution:
        count = max(1, int(hinglish_threads * ratio))
        hin_plan.extend([(thread_type, "hin")] * count)
    random.shuffle(hin_plan)

    print(f"\n🇮🇳 Pass 1: Generating {len(hin_plan)} Hinglish supplement threads...")
    print("   Distribution: 20% benign, 30% mild, 20% subtle, 20% escalating, 10% toxic")
    for thread_type, lang in tqdm(hin_plan):
        is_toxic_seed = thread_type in ("toxic", "escalating")
        seed_texts = get_seed_texts(df, "hi", toxic=is_toxic_seed, n=5)  # Hindi seeds for Hinglish
        thread = generate_thread(seed_texts, thread_type, lang, system_prompt=SYSTEM_PROMPT_HINGLISH)
        if thread:
            thread["thread_id"] = _unique_id(lang, thread_type, counter)
            thread["language"] = "hin"
            new_threads.append(thread)
        else:
            failed += 1
        if (len(new_threads) + failed) % 20 == 0:
            time.sleep(0.3)

    # ── Pass 2: Mid-toxicity across all languages ────────────
    mid_tox_distribution = [
        ("mild",       0.50),
        ("escalating", 0.30),
        ("subtle",     0.20),
    ]
    mid_plan = []
    for lang in all_languages:
        for thread_type, ratio in mid_tox_distribution:
            count = max(1, int(mid_tox_threads_per_lang * ratio))
            mid_plan.extend([(thread_type, lang)] * count)
    random.shuffle(mid_plan)

    print(f"\n🌍 Pass 2: Generating {len(mid_plan)} mid-toxicity threads across all languages...")
    print("   Distribution: 50% mild, 30% escalating, 20% subtle")
    for thread_type, lang in tqdm(mid_plan):
        is_toxic_seed = thread_type == "escalating"
        sys_prompt = SYSTEM_PROMPT_HINGLISH if lang == "hin" else SYSTEM_PROMPT
        seed_texts = get_seed_texts(df, lang, toxic=is_toxic_seed, n=5)
        thread = generate_thread(seed_texts, thread_type, lang, system_prompt=sys_prompt)
        if thread:
            thread["thread_id"] = _unique_id(lang, thread_type, counter)
            thread["language"] = lang
            new_threads.append(thread)
        else:
            failed += 1
        if (len(new_threads) + failed) % 50 == 0:
            time.sleep(0.5)

    # ── Stats ─────────────────────────────────────────────────
    print(f"\n✅ Generated {len(new_threads)} new threads ({failed} failed/rejected)")
    counts = Counter(t.get("thread_type") for t in new_threads)
    print("   Type breakdown:", dict(sorted(counts.items())))
    lang_counts = Counter(t.get("language") for t in new_threads)
    print("   Language breakdown:", dict(sorted(lang_counts.items())))

    # ── Append and save ───────────────────────────────────────
    all_threads = existing_threads + new_threads
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_threads, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(all_threads)} total threads → {output_file}")
    print(f"   ({len(existing_threads)} existing + {len(new_threads)} new)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Discord threads via DeepSeek V3")
    parser.add_argument(
        "--supplement",
        action="store_true",
        help="Append supplement threads (Hinglish + mid-toxicity) to existing causal_threads.json",
    )
    parser.add_argument("--hinglish-threads", type=int, default=40,
                        help="Number of Hinglish supplement threads (default: 40)")
    parser.add_argument("--mid-tox-per-lang", type=int, default=15,
                        help="Mid-toxicity threads per language in pass 2 (default: 15)")
    args = parser.parse_args()

    if args.supplement:
        generate_supplement_threads(
            hinglish_threads=args.hinglish_threads,
            mid_tox_threads_per_lang=args.mid_tox_per_lang,
        )
    else:
        generate_causal_threads(target_threads=800)
