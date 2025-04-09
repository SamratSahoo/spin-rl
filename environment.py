import imageio
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen import (
    MujocoHandPenEnv,
)
from utils import angles_to_quaternion, quaternion_to_angles


class WrappedPenSpinEnv(MujocoHandPenEnv):
    def __init__(
        self,
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=500,
        temp_dir="./tmp",
        render_overlay=True,
        target_rotation_speed=0.1
    ):
        super().__init__(
            render_mode=render_mode,
            reward_type=reward_type,
            target_position="ignore",
        )
        self.recording = False
        self.frames = []
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.temp_dir = temp_dir
        self.render_overlay = render_overlay
        self.target_rotation_speed = target_rotation_speed

        os.makedirs(self.temp_dir, exist_ok=True)
        self.init_target_coords()

    def init_target_coords(self):
        coords = np.concatenate([self.get_pen_coords()[:3], angles_to_quaternion(0, np.pi/2, 0)])
        self.set_target_coords(coords)
        self.fixed_target_pos = coords[:3]
    
    def get_pen_coords(self):
        return self._utils.get_joint_qpos(self.model, self.data, "object:joint")
    
    def get_target_coords(self):
        return self._utils.get_joint_qpos(self.model, self.data, "target:joint")

    def set_target_coords(self, coords):
        self._utils.set_joint_qpos(self.model, self.data, "target:joint", coords)
    
    def get_rgb_frame(self):
        frame = self.mujoco_renderer.render("rgb_array")
        
        if self.render_overlay:
            pen_coords = self.get_pen_coords()
            target_coords = self.get_target_coords()

            frame = self._add_text_overlay(frame, pen_coords, target_coords)

        return frame
    
    def _add_text_overlay(self, frame, pen_coords, target_coords):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        text = f"Pen Pos: x={pen_coords[0]:.3f}, y={pen_coords[1]:.3f}, z={pen_coords[2]:.3f}"
        text2 = f"Pen Quat: {pen_coords[3]:.3f}, {pen_coords[4]:.3f}, {pen_coords[5]:.3f}, {pen_coords[6]:.3f}"
        text3 = f"Target Pos: x={target_coords[0]:.3f}, y={target_coords[1]:.3f}, z={target_coords[2]:.3f}"
        text4 = f"Target Quat: {target_coords[3]:.3f}, {target_coords[4]:.3f}, {target_coords[5]:.3f}, {target_coords[6]:.3f}"
        
        draw.text((10, 10), text, fill=(255, 0, 0))
        draw.text((10, 30), text2, fill=(255, 0, 0))
        draw.text((10, 50), text3, fill=(0, 0, 255))
        draw.text((10, 70), text4, fill=(0, 0, 255))
        
        return np.array(img)

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

        self.rotate_target()

        self.current_step += 1
        if self.recording:
            self.frames.append(self.get_rgb_frame())

        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            truncated = True
            terminated = True
        else:
            # Not sure if the environment inherently truncates so explicitly setting it
            truncated = False
            terminated = False

        return obs, reward, terminated, truncated, info

    def sample(self):
        return self.action_space.sample()
    
    def rotate_target(self):
        target_coords = self.get_target_coords()
        target_pos = self.fixed_target_pos  
        target_rot = target_coords[3:]
        
        roll, pitch, yaw = quaternion_to_angles(*target_rot)

        yaw += self.target_rotation_speed
        new_rot = angles_to_quaternion(roll, pitch, yaw)
        self.set_target_coords(np.concatenate((target_pos, new_rot)))
