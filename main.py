import json
import os
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from algorithms.sb3 import StableBaseLines3Runner
from environment import WrappedPenSpinEnv
import gymnasium as gym

import time


def current_milli_time():
    return round(time.time() * 1000)


def make_env(reward_type, temp_dir):
    def _init():
        env = WrappedPenSpinEnv(
            reward_type=reward_type, temp_dir=temp_dir, max_episode_steps=300
        )
        env.reset()
        return env

    return _init


def train_sb3(
    algorithm,
    timesteps,
    policy,
    parallel_envs,
    model_settings,
    eval_recording_folder,
    eval_freq,
    model_save_folder,
    train_from_checkpoint,
    reward_type,
    temp_dir,
):
    train_env_fns = [
        make_env(reward_type=reward_type, temp_dir=temp_dir)
        for j in range(parallel_envs)
    ]

    train_env = SubprocVecEnv(train_env_fns)
    train_env.seed(seed=42)

    eval_env = WrappedPenSpinEnv(max_episode_steps=300)
    eval_env.reset()

    sb3_runner = StableBaseLines3Runner(
        env=train_env,
        eval_env=eval_env,
        best_model_save_folder=model_save_folder,
        best_model_name=f"{algorithm}_best_model.zip",
        eval_freq=eval_freq,
        eval_recording_folder=eval_recording_folder,
        algorithm=algorithm,
        checkpoint=(
            f"{model_save_folder}/{algorithm}_best_model"
            if train_from_checkpoint
            else None
        ),
    )
    sb3_runner.train(policy=policy, timesteps=timesteps, model_settings=model_settings)

    model_recording = f"{eval_recording_folder}/{algorithm}_final"
    model_path = f"{os.path.abspath(model_save_folder)}/{algorithm}_best_model"

    sb3_runner.run_evaluation(
        recording=f"{model_recording}.mp4",
        model_path=model_path,
    )

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different RL algorithms on the Pen Spin Environment."
    )
    parser.add_argument(
        "--model_config", help="The model configuration file", default="sac"
    )
    parser.add_argument(
        "--env_config", help="The environment configuration file", default="env"
    )
    args = parser.parse_args()

    with open(f"./config/{args.model_config}.json", "r") as f:
        model_config = json.load(f)

    with open(f"./config/{args.env_config}.json", "r") as f:
        env_config = json.load(f)

    model_algorithm = model_config["algorithm"]
    model_timesteps = model_config["timesteps"]
    model_policy = model_config["policy"]
    model_settings = model_config["model"]
    model_eval_recording_folder = model_config["eval_recording_folder"]
    model_save_folder = model_config["model_save_folder"]
    model_eval_freq = model_config["eval_freq"]
    model_train_from_checkpoint = model_config["train_from_checkpoint"]
    model_temp_dir = model_config["temp_dir"]

    env_parallel_envs = env_config["parallel_envs"]
    env_reward_type = env_config["reward_type"]

    train_sb3(
        algorithm=model_algorithm,
        timesteps=model_timesteps,
        policy=model_policy,
        model_settings=model_settings,
        eval_recording_folder=model_eval_recording_folder,
        model_save_folder=model_save_folder,
        eval_freq=model_eval_freq,
        train_from_checkpoint=model_train_from_checkpoint,
        parallel_envs=env_parallel_envs,
        reward_type=env_reward_type,
        temp_dir=model_temp_dir,
    )

    # env = WrappedPenSpinEnv(render_mode="rgb_array", max_episode_steps=300)
    # current_state, _ = env.reset()
    # current_step = 1
    # done = False
    # env.start_recording()
    # while not done:
    #     action = env.sample()
    #     next_state, reward, done, truncated, _ = env.step(action)
    #     current_state = next_state
    #     current_step += 1
    # env.stop_recording()
    # env.close()
