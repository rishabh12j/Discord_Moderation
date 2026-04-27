import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.language_detector import detect_language

def construct_episode_ledger(input_file: str = "data/processed/context_strings.json",
                             output_file: str = "data/processed/episodes.json"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}.")

    with open(input_file, 'r', encoding='utf-8') as f:
        states = json.load(f)

    episodes = {}

    print(f"Mapping {len(states)} steps and running language detection...")
    
    for state in states:
        thread_id = state.get("thread_id")
        step_index = state.get("global_step_index")
        user_id = state.get("user_id")
        raw_text = state.get("raw_text", "")
        
        lang_result = detect_language(raw_text)
        language = lang_result.lang

        if thread_id not in episodes:
            episodes[thread_id] = {
                "thread_id": thread_id,
                "step_indices": [],
                "user_ids": [],
                "languages": []
            }
        
        episodes[thread_id]["step_indices"].append(step_index)
        episodes[thread_id]["user_ids"].append(user_id)
        episodes[thread_id]["languages"].append(language)

    episode_list = list(episodes.values())

    # Verify structural integrity
    for ep in episode_list:
        if len(ep["step_indices"]) != len(ep["user_ids"]):
            raise ValueError(f"Index mismatch in thread {ep['thread_id']}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(episode_list, f, indent=4)

    print("Episode state construction complete.")
    print(f"Mapped {len(states)} global steps into {len(episode_list)} discrete episodes.")
    print(f"Output saved to {output_file}.")

if __name__ == "__main__":
    construct_episode_ledger()