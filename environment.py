import imageio
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen import (
    MujocoHandPenEnv,
)
from utils import angles_to_quaternion, quaternion_to_angles, quat_from_angle_and_axis


class WrappedPenSpinEnv(MujocoHandPenEnv):
    def __init__(
        self,
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=500,
        temp_dir="./tmp",
        render_overlay=True,
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
        self.prev_pen_coords = None
        self.accumulated_reward = 0.0

        os.makedirs(self.temp_dir, exist_ok=True)
            
    def get_pen_coords(self):
        return self._utils.get_joint_qpos(self.model, self.data, "object:joint")
    
    def set_pen_coords(self, coords):
        self._utils.set_joint_qpos(self.model, self.data, "object:joint", coords)
    
    def get_rgb_frame(self):
        frame = self.mujoco_renderer.render("rgb_array")
        
        if self.render_overlay:
            pen_coords = self.get_pen_coords()

            frame = self._add_text_overlay(frame, pen_coords)

        return frame
    
    def _add_text_overlay(self, frame, pen_coords):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        text = f"Pen Pos: x={pen_coords[0]:.3f}, y={pen_coords[1]:.3f}, z={pen_coords[2]:.3f}"
        text2 = f"Pen Quat: {pen_coords[3]:.3f}, {pen_coords[4]:.3f}, {pen_coords[5]:.3f}, {pen_coords[6]:.3f}"
        text5 = f"Accumulated Reward: {self.accumulated_reward:.3f}"
        
        draw.text((10, 10), text, fill=(255, 0, 0))
        draw.text((10, 30), text2, fill=(255, 0, 0))
        draw.text((10, 50), text5, fill=(0, 128, 0))
        
        return np.array(img)

    def render(self):
        return self.get_rgb_frame()
    
    def start_recording(self):
        self.recording = True
        self.frames = []

    def stop_recording(self, recording_name="runs/eval/random_policy.mp4"):
        if self.recording and self.frames:
            self._save_video(recording_name)

        self.recording = False
        self.frames = []

    def _save_video(self, recording_name):
        imageio.mimsave(recording_name, self.frames, fps=30)

    def calc_reward(self):
        if self.prev_pen_coords is None:
            return 0.0

        prev_rot = self.prev_pen_coords[3:]
        curr_rot = self.get_pen_coords()[3:]

        _, _, prev_yaw = quaternion_to_angles(*prev_rot)
        roll, pitch, curr_yaw = quaternion_to_angles(*curr_rot)

        delta_yaw = curr_yaw - prev_yaw
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        
        alpha = 5.0
        spin_reward = delta_yaw * alpha 
        
        drop_penalty = -5 if self.get_pen_coords()[2] < 0.0 else 0.0
        
        beta = 0.2
        ideal_pitch = np.pi/2
        stability_reward = beta * (1.0 - min(abs(pitch - ideal_pitch), abs(roll)) / np.pi)
        
        height = self.get_pen_coords()[2]
        optimal_height = 0.2
        height_reward = 0.1 * (1.0 - min(abs(height - optimal_height), 0.1) * 10.0)
        
        min_rotation_speed = 0.1  # Minimum acceptable rotation per step
        stationary_penalty = -2.0 if abs(delta_yaw) < min_rotation_speed else 0.0
        
        total_reward = spin_reward + stability_reward + height_reward + drop_penalty + stationary_penalty
        return total_reward

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        reward = self.calc_reward()
        self.accumulated_reward += reward

        self.current_step += 1
        if self.recording:
            self.frames.append(self.get_rgb_frame())

        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            self.accumulated_reward = 0
            truncated = True
            terminated = True
        else:
            truncated = False
            terminated = False

        self.prev_pen_coords = self.get_pen_coords()
        return obs, reward, terminated, truncated, info

    def sample(self):
        return self.action_space.sample()

class GeneralizedPenSpinEnv(MujocoHandPenEnv):
    def __init__(
        self,
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=500,
        temp_dir="./tmp",
        render_overlay=True,
    ):
        super().__init__(
            render_mode=render_mode,
            reward_type=reward_type,
            target_position="ignore",
            rotation_threshold=0.2,
        )
        self.recording = False
        self.frames = []
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.temp_dir = temp_dir
        self.render_overlay = render_overlay
        self.prev_pen_coords = None
        self.accumulated_reward = 0.0

        os.makedirs(self.temp_dir, exist_ok=True)
            
    def get_pen_coords(self):
        return self._utils.get_joint_qpos(self.model, self.data, "object:joint")
    
    def set_pen_coords(self, coords):
        self._utils.set_joint_qpos(self.model, self.data, "object:joint", coords)
    
    def get_rgb_frame(self):
        frame = super().render()
        
        if self.render_overlay:
            pen_coords = self.get_pen_coords()
            target_coords = self.goal
            _, goal_distance = self._goal_distance(pen_coords, target_coords)
            frame = self._add_text_overlay(frame, pen_coords, target_coords, goal_distance)
        return frame
    
    def _add_text_overlay(self, frame, pen_coords, target_coords, goal_distance):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        text = f"Pen Pos: x={pen_coords[0]:.3f}, y={pen_coords[1]:.3f}, z={pen_coords[2]:.3f}"
        text2 = f"Pen Quat: {pen_coords[3]:.3f}, {pen_coords[4]:.3f}, {pen_coords[5]:.3f}, {pen_coords[6]:.3f}"
        text3 = f"Target Pos: x={target_coords[0]:.3f}, y={target_coords[1]:.3f}, z={target_coords[2]:.3f}"
        text4 = f"Target Quat: {target_coords[3]:.3f}, {target_coords[4]:.3f}, {target_coords[5]:.3f}, {target_coords[6]:.3f}"
        text5 = f"Accumulated Reward: {self.accumulated_reward:.3f}"
        text6 = f"Goal Distance: {goal_distance:.3f}"
        
        draw.text((10, 10), text, fill=(255, 0, 0))
        draw.text((10, 30), text2, fill=(255, 0, 0))
        draw.text((10, 50), text3, fill=(0, 0, 255))
        draw.text((10, 70), text4, fill=(0, 0, 255))
        draw.text((10, 90), text5, fill=(0, 0, 255))
        draw.text((10, 110), text6, fill=(0, 0, 255))
        
        return np.array(img)

    def render(self):
        return self.get_rgb_frame()
    
    def start_recording(self):
        self.recording = True
        self.frames = []

    def stop_recording(self, recording_name="runs/eval/random_policy.mp4"):
        if self.recording and self.frames:
            self._save_video(recording_name)

        self.recording = False
        self.frames = []

    def _save_video(self, recording_name):
        imageio.mimsave(recording_name, self.frames, fps=30)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        pen_coords = self.get_pen_coords()
        target_coords = self.goal

        # self.set_pen_coords(target_coords)
    
        if self._is_success(pen_coords, target_coords):
            self.randomize_target_coords()
            reward += 10
        
        self.accumulated_reward += reward

        self.current_step += 1
        if self.recording:
            self.frames.append(self.get_rgb_frame())

        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            self.accumulated_reward = 0
            truncated = True
            terminated = True
        else:
            truncated = False
            terminated = False            

        self.prev_pen_coords = self.get_pen_coords()
        return obs, reward, terminated, truncated, info

    def sample(self):
        return self.action_space.sample()

    def _is_success(self, achieved_goal, desired_goal):
        _, d_rot = self._goal_distance(achieved_goal, desired_goal)
        return d_rot < self.rotation_threshold

    def randomize_target_coords(self):
        angle = self.np_random.uniform(-np.pi, np.pi)
        axis = self.np_random.uniform(-1, 1, size=3)
        target_quat = quat_from_angle_and_axis(angle, axis)

        curr_target = self.goal
        self.goal = np.concatenate((curr_target[:3], target_quat))
    
    def _goal_distance(self, achieved_goal, desired_goal):
        d_pos, d_rot = super()._goal_distance(achieved_goal, desired_goal)
        d_rot = (d_rot + np.pi) % (2 * np.pi) - np.pi
        return d_pos, d_rot