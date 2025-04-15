import os
from environment import WrappedPenSpinEnv
import os
import gymnasium as gym
import time
import torch
from algorithms.ppo import Agent
from algorithms.sac import Actor

os.environ['MUJOCO_GL'] = 'osmesa'

def run_random_eval():
    base_env = WrappedPenSpinEnv(
        reward_type="dense", temp_dir=f"./tmp/eval", 
                max_episode_steps=300
    )
    env = gym.wrappers.RecordVideo(base_env, f"runs/spin_rl__eval__{int(time.time())}/videos")
    env.recorded_frames = []
    current_state, _ = env.reset()

    while True:
        action = base_env.sample()
        next_state, reward, terminate, truncate, info = env.step(action)
        if terminate or truncate:
            break
        current_state = next_state
    env.close()

def run_model_eval(model_path: str, Model: torch.nn.Module):
    base_env = WrappedPenSpinEnv(
        reward_type="dense", temp_dir=f"./tmp/eval", 
                max_episode_steps=300
    )
    env = gym.wrappers.RecordVideo(base_env, f"runs/spin_rl__eval__{int(time.time())}/videos")
    gym.vector.SyncVectorEnv(
            [env],
            autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
        )

    agent = Model(env)
    agent.load_state_dict(torch.load(model_path))
    env.recorded_frames = []
    current_state, _ = env.reset()

    while True:
        action = agent.get_action_and_value(current_state)
        next_state, reward, terminate, truncate, info = env.step(action)
        if terminate or truncate:
            break
        current_state = next_state
    env.close()

if __name__ == "__main__":
    run_random_eval()

    # For SAC
    # run_model_eval("your-model-path", Actor)

    # For PPO
    # run_model_eval("your-model-path", Agent)
