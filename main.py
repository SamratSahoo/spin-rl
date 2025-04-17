import os
import argparse
from environment import WrappedPenSpinEnv
import os
import time
from algorithms import ppo, sac, ddpg, vpg

os.environ['MUJOCO_GL'] = 'osmesa'

def make_env(reward_type, temp_dir):
    def _init():
        env = WrappedPenSpinEnv(
            reward_type=reward_type, temp_dir=temp_dir, max_episode_steps=300
        )
        env.reset()
        return env

    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different RL algorithms on the Pen Spin Environment."
    )
    parser.add_argument(
        "--algorithm", help="The algorithm", default="sac"
    )
    args = parser.parse_args()

    if args.algorithm == "ppo":
        trainer = ppo.PPOTrainer(env_type=WrappedPenSpinEnv)
    elif args.algorithm == "sac":
        trainer = sac.SACTrainer(env_type=WrappedPenSpinEnv)
    elif args.algorithm == "ddpg":
        trainer = ddpg.DDPGTrainer(env_type=WrappedPenSpinEnv)
    elif args.algorithm == "vpg":
        trainer = vpg.VPGTrainer(env_type=WrappedPenSpinEnv)

    trainer.train()