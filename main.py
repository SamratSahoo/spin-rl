import os
import argparse
from environment import WrappedPenSpinEnv, GeneralizedPenSpinEnv, GeneralizedPenSpinEnvV2
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

    if args.environment == "general":
        env_id = "gen_spin_rl"
        env_type = GeneralizedPenSpinEnv
        goal_size = 4
    elif args.environment == "general-v2":
        env_id = "gen_v2_spin_rl"
        env_type = GeneralizedPenSpinEnvV2
        goal_size = 7
    else:
        env_id = "spin_rl"
        env_type = WrappedPenSpinEnv
        goal_size = 0
    
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