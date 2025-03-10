import sys
import os
from pathlib import Path
from algorithms.sb3 import StableBaseLines3Runner
from environment import WrappedPenSpinEnv
import os

os.environ['MUJOCO_GL'] = 'osmesa'

def run_simple_eval(algorithm="sac"):
    env = WrappedPenSpinEnv(temp_dir="./tmp/eval", max_episode_steps=300)
    env.reset()
    sb3_instance = StableBaseLines3Runner(env=env, eval_env=env, algorithm=algorithm)
    sb3_instance.run_evaluation(
        recording=f"./recordings/{algorithm}/evaluate.mp4",
        model_path=f"{os.getcwd()}/models/{algorithm}/{algorithm}_best_model",
    )

if __name__ == "__main__":
    run_simple_eval()
