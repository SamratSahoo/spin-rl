import imageio
import os
import numpy as np
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen import (
    MujocoHandPenEnv,
)


class WrappedPenSpinEnv(MujocoHandPenEnv):
    def __init__(
        self,
        render_mode="human",
        reward_type="dense",
        max_episode_steps=500,
        chunk_size=500,
    ):
        super().__init__(
            render_mode=render_mode,
            reward_type=reward_type,
        )
        self.recording = False
        self.frames = []
        self.chunk_size = chunk_size
        self.output_dir = "recordings"
        self.temp_chunks = []
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        os.makedirs(self.output_dir, exist_ok=True)

    def get_rgb_frame(self):
        return self.mujoco_renderer.render("rgb_array")

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.temp_chunks = []

    def stop_recording(self, recording_name="random_policy.mp4"):
        if self.recording and self.frames:
            self._save_video(recording_name)
        self._combine_chunks()
        self.recording = False
        self.frames = []
        self.temp_chunks = []

    def _save_video(self, recording_name):
        file_path = os.path.join(self.output_dir, recording_name)
        imageio.mimsave(file_path, self.frames, fps=30)
        self.temp_chunks.append(file_path)

    def _combine_chunks(self):
        if len(self.temp_chunks) > 1:
            combined_frames = []
            for chunk in self.temp_chunks:
                reader = imageio.get_reader(chunk, format="mp4")
                combined_frames.extend([frame for frame in reader])
                reader.close()
            combined_file = os.path.join(self.output_dir, "final_recording.mp4")
            imageio.mimsave(combined_file, combined_frames, fps=30)
            print(f"Combined video saved: {combined_file}")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1
        if self.recording:
            self.frames.append(self.get_rgb_frame())
            if len(self.frames) >= self.chunk_size:
                self._save_video()
                self.frames = []

        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            terminated = True
            truncated = True
        return obs, reward, terminated, truncated, info

    def sample(self):
        return self.action_space.sample()
