import os
import argparse
from environment import WrappedPenSpinEnv, GeneralizedPenSpinEnv
import os
import time
from algorithms import ppo, sac, ddpg, vpg

os.environ['MUJOCO_GL'] = 'osmesa'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run different RL algorithms on the Pen Spin Environment."
    )
    parser.add_argument(
        "--algorithm", help="The algorithm", default="sac"
    )
    parser.add_argument(
        "--environment", help="The env", default="general"
    )
    args = parser.parse_args()
    env_type = GeneralizedPenSpinEnv if args.environment == "general" else WrappedPenSpinEnv
    goal_size = 4 if env_type == GeneralizedPenSpinEnv else 0
    env_id = "gen_spin_rl" if env_type == GeneralizedPenSpinEnv else "spin_rl"
    
    if args.algorithm == "ppo":
        trainer = ppo.PPOTrainer(
            env_type=env_type,
            goal_size=goal_size,
            env_id=env_id
        )
    elif args.algorithm == "sac":
        trainer = sac.SACTrainer(env_type=env_type,
                                 goal_size=goal_size,
                                 env_id=env_id)
    elif args.algorithm == "ddpg":
        trainer = ddpg.DDPGTrainer(env_type=env_type,
                                   goal_size=goal_size,
                                   env_id=env_id)
    elif args.algorithm == "vpg":
        trainer = vpg.VPGTrainer(env_type=env_type, 
                                 goal_size=goal_size,
                                 env_id=env_id)

    trainer.train()