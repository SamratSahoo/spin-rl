from stable_baselines3 import SAC
from algorithms.utils import make_env
import gymnasium as gym
import time
import os
from stable_baselines3.common.vec_env import SubprocVecEnv

class SB3SACTrainer:
    def __init__(self, env_type, seed=1, num_envs=4,env_id="spin_rl", exp_name=os.path.basename(__file__)[: -len(".py")]) -> None:
        self.num_envs = num_envs
        self.env_id = env_id
        self.exp_name = exp_name
        self.seed = seed
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        self.envs = SubprocVecEnv(
            [make_env(env_type, self.env_id, i, True, self.run_name) for i in range(self.num_envs)]
        )
        self.model = SAC("MultiInputPolicy", self.envs, verbose=1, tensorboard_log=f"runs/{self.run_name}")

    def train(self, total_timesteps=10000000):
        self.model.learn(total_timesteps=total_timesteps, log_interval=4)
        self.model.save(f"runs/{self.run_name}/actor")
