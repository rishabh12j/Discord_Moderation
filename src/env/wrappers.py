import gymnasium as gym
from typing import Any, Dict, Tuple
import numpy as np

class LagrangianRewardWrapper(gym.Wrapper):
    """
    Gym wrapper that implements a Lagrangian multiplier for Constrained RL.
    Modifies the reward by subtracting (lambda_val * safety_cost).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Lambda starts at 0. The callback (Day 25) will increase it if constraints are violated.
        self.lambda_val = 0.0

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Take a step in the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract the safety cost we logged in Day 15
        cost = info.get("safety_cost", 0.0)
        
        # Apply the Lagrangian penalty
        lagrangian_reward = reward - (self.lambda_val * cost)
        
        # We can also log the unmodified reward and the current lambda for W&B tracking later
        info["base_reward"] = reward
        info["lagrangian_lambda"] = self.lambda_val
        
        return obs, lagrangian_reward, terminated, truncated, info

    def update_lambda(self, delta: float) -> float:
        """
        Increases the lambda multiplier. 
        Called dynamically during training if the average cost exceeds our threshold.
        """
        self.lambda_val += delta
        return self.lambda_val