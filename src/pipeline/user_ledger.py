import os
import json
import random
import ast

def initialize_user_ledgers(input_file: str = "data/processed/causal_threads.json",
                            output_file: str = "data/processed/user_ledgers.json",
                            seed: int = 42):
    """
    Extracts unique mock users from causal threads and generates baseline state ledgers.
    Includes parsing logic to handle stringified JSON from local LLMs.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    with open(input_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)

    unique_users = set()
    
    for thread in threads:
        if not isinstance(thread, dict):
            continue
            
        messages = thread.get("messages", [])
        
        if isinstance(messages, dict):
            messages = [messages]
            
        if not isinstance(messages, list):
            continue

        for message in messages:
            if isinstance(message, str):
                try:
                    message = ast.literal_eval(message)
                except (ValueError, SyntaxError):
                    continue
            
            if isinstance(message, dict):
                user_id = message.get("user_id")
                if user_id and isinstance(user_id, str):
                    unique_users.add(user_id)

    if not unique_users:
        raise ValueError("No valid users could be extracted. Verify LLM JSON outputs.")

    random.seed(seed)
    
    ledgers = {}
    for user_id in unique_users:
        is_offender = random.random() < 0.15 
        
        historical_toxicity_avg = random.uniform(0.4, 0.9) if is_offender else random.uniform(0.0, 0.2)
        warn_count = random.randint(1, 4) if is_offender else 0
        account_age_days = random.randint(1, 3650) 

        ledgers[user_id] = {
            "historical_toxicity_avg": round(historical_toxicity_avg, 4),
            "warn_count": warn_count,
            "account_age_days": account_age_days
        }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ledgers, f, indent=4)

    print(f"Generated baseline states for {len(unique_users)} unique users.")
    print(f"Saved to {output_file}.")

if __name__ == "__main__":
    initialize_user_ledgers()