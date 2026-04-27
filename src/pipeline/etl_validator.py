import os
import json
import ast

def validate_pipeline_state(threads_file: str = "data/processed/causal_threads.json",
                            ledgers_file: str = "data/processed/user_ledgers.json",
                            output_file: str = "data/processed/clean_threads.json"):
    """
    Validates the structural integrity of the Phase 1 ETL outputs.
    Sanitizes stringified LLM outputs, enforces minimum sequence lengths, 
    and checks referential integrity between threads and ledgers.
    """
    if not os.path.exists(threads_file):
        raise FileNotFoundError(f"Missing {threads_file}")
    if not os.path.exists(ledgers_file):
        raise FileNotFoundError(f"Missing {ledgers_file}")

    with open(threads_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)

    with open(ledgers_file, 'r', encoding='utf-8') as f:
        ledgers = json.load(f)

    clean_threads = []
    dropped_count = 0

    for thread in threads:
        if not isinstance(thread, dict):
            dropped_count += 1
            continue
            
        raw_messages = thread.get("messages", [])
        
        # Normalize unexpected top-level structures from the LLM
        if isinstance(raw_messages, dict):
            raw_messages = [raw_messages]
        elif isinstance(raw_messages, str):
            try:
                raw_messages = ast.literal_eval(raw_messages)
                if not isinstance(raw_messages, list):
                    raw_messages = [raw_messages]
            except (ValueError, SyntaxError):
                dropped_count += 1
                continue

        if not isinstance(raw_messages, list):
            dropped_count += 1
            continue

        parsed_messages = []
        for msg in raw_messages:
            # Parse stringified dictionaries within the list
            if isinstance(msg, str):
                try:
                    msg = ast.literal_eval(msg)
                except (ValueError, SyntaxError):
                    continue 
            
            # Keep only strictly valid message objects
            if isinstance(msg, dict) and "user_id" in msg and "text" in msg:
                parsed_messages.append(msg)

        # Enforce minimum sequence length for momentum calculations
        if len(parsed_messages) < 3:
            dropped_count += 1
            continue

        # Verify referential integrity against the ledger
        valid_thread = True
        for msg in parsed_messages:
            user_id = msg.get("user_id")
            if not user_id or user_id not in ledgers:
                valid_thread = False
                break
        
        if valid_thread:
            # Overwrite the dirty structures with the sanitized objects
            thread["messages"] = parsed_messages
            clean_threads.append(thread)
        else:
            dropped_count += 1

    if not clean_threads:
        raise ValueError("Pipeline failure: 0 threads passed validation. All sequences were invalid or lacked ledgers.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_threads, f, indent=4)

    print(f"ETL Validation Complete.")
    print(f"Passed: {len(clean_threads)} threads.")
    print(f"Dropped: {dropped_count} threads (Failed parsing, length, or ledger constraints).")
    print(f"Clean, normalized data saved to {output_file}.")

if __name__ == "__main__":
    validate_pipeline_state()