import os
import json

def build_context_windows(input_file: str = "data/processed/clean_threads.json",
                          output_file: str = "data/processed/context_strings.json",
                          k: int = 3):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}.")

    with open(input_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)

    pad_token = "[PAD]"
    sep_token = " [SEP] "
    
    contextual_states = []
    global_step_index = 0

    for thread in threads:
        thread_id = thread.get("thread_id")
        messages = thread.get("messages", [])
        
        window = [pad_token] * k

        for msg_index, msg in enumerate(messages):
            raw_text = msg.get("text", "").strip()
            user_id = msg.get("user_id")

            window.append(raw_text)
            if len(window) > k + 1:
                window.pop(0)

            context_string = sep_token.join(window)

            contextual_states.append({
                "global_step_index": global_step_index,
                "thread_id": thread_id,
                "thread_step": msg_index,
                "user_id": user_id,
                "raw_text": raw_text,
                "context_string": context_string
            })
            global_step_index += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contextual_states, f, indent=4)

    print(f"Generated {len(contextual_states)} sequential contextual states.")
    print(f"Output saved to {output_file}.")

if __name__ == "__main__":
    build_context_windows()