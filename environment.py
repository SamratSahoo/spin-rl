import numpy as np
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen import (
    MujocoHandPenEnv,
)


class WrappedPenSpinEnv(MujocoHandPenEnv):
    def __init__(self, render_mode="human", reward_type="dense"):
        super().__init__(render_mode=render_mode, reward_type=reward_type)

    def sample(self):
        return self.action_space.sample()
