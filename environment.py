import imageio
import os
import numpy as np
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen import (
    MujocoHandPenEnv,
)


class WrappedPenSpinEnv(MujocoHandPenEnv):
    def __init__(
        self,
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=500,
        temp_dir="./tmp",
    ):
        super().__init__(
            render_mode=render_mode,
            reward_type=reward_type,
        )
        self.recording = False
        self.frames = []
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_rgb_frame(self):
        return self.mujoco_renderer.render("rgb_array")

    def start_recording(self):
        self.recording = True
        self.frames = []

    def stop_recording(self, recording_name="random_policy.mp4"):
        if self.recording and self.frames:
            self._save_video(recording_name)

        self.recording = False
        self.frames = []

    def _save_video(self, recording_name):
        imageio.mimsave(recording_name, self.frames, fps=30)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1
        if self.recording:
            self.frames.append(self.get_rgb_frame())

        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            terminated = True
            truncated = True
        return obs, reward, terminated, truncated, info

    def sample(self):
        return self.action_space.sample()
