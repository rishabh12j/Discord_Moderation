import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from collections import defaultdict
from env.discord_env import DiscordEnv
from env.wrappers import LagrangianRewardWrapper
from utils.toxicity_judge import ToxicityJudge
from utils.language_utils import detect_language

def run_loop_test():
    print("Initializing environment...")
    base_env = DiscordEnv()
    env = LagrangianRewardWrapper(base_env)
    
    obs, info = env.reset()
    
    action_counts_per_lang = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    total_steps = 1000
    
    print(f"Running {total_steps} random steps...")
    for step in range(total_steps):
        # Use our custom action masks to only take valid random actions
        valid_actions = np.where(env.action_masks())[0]
        action = np.random.choice(valid_actions)
        
        # We access the private language variable just for logging/testing, NOT for the agent
        lang = env.env._current_language 
        action_counts_per_lang[lang][action] += 1
        
        obs, reward, term, trunc, info = env.step(action)
        
        if term or trunc:
            obs, info = env.reset()

    print("\n=== Action Distribution by Language (1000 steps) ===")
    print(f"{'Lang':<6} | {'ALLOW':<6} | {'WARN':<6} | {'DELETE':<6} | {'BAN':<6}")
    print("-" * 40)
    for lang, counts in sorted(action_counts_per_lang.items()):
        print(f"{lang:<6} | {counts[0]:<6} | {counts[1]:<6} | {counts[2]:<6} | {counts[3]:<6}")

def run_edge_cases():
    print("\n=== Testing Edge Cases ===")
    judge = ToxicityJudge()
    
    edge_cases = [
        "",                             # Empty string
        "😊😂🔥",                       # Emoji only
        "Hello amigo, como estas?",     # Mixed language (Code-switching)
    ]
    
    for text in edge_cases:
        lang_res = detect_language(text)
        score = judge.score_text(text)
        print(f"Text: {repr(text):<30} | Lang: {lang_res.lang:<8} | Tox Score: {score:.3f}")

if __name__ == "__main__":
    run_loop_test()
    run_edge_cases()