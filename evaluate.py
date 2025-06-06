import os
from environment import WrappedPenSpinEnv, GeneralizedPenSpinEnv
import os
import gymnasium as gym
import time
import torch
from algorithms.ppo import Agent as PPOActor
from algorithms.sac import Actor as SACActor
from algorithms.ddpg import Actor as DDPGActor

os.environ['MUJOCO_GL'] = 'osmesa'

def run_random_eval(env_type=WrappedPenSpinEnv):
    base_env = env_type(
        reward_type="dense", temp_dir=f"./tmp/eval", 
                max_episode_steps=300
    )
    curr_time = int(time.time())
    env = gym.wrappers.RecordVideo(base_env, f"runs/spin_rl__eval__{curr_time}/videos")
    print(f"Video will be saved in: runs/spin_rl__eval__{curr_time}/videos")
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
    run_random_eval(env_type=GeneralizedPenSpinEnv)

    # For SAC
    # run_model_eval("your-model-path", SACActor)

    # For PPO
    # run_model_eval("your-model-path", PPOActor)

    # For DDPG
    # run_model_eval("your-model-path", DDPGActor)

