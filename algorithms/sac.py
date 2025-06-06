# Implementation Based on Clean RL: https://github.com/vwxyzjn/cleanrl
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from algorithms.evaluate_agent import evaluate
from algorithms.utils import make_env, transform_obs
from gymnasium import spaces

GOAL_SIZE = 4
# Note inputs to nets here is observation + goal, not just observation
class SoftQNetwork(nn.Module):
    def __init__(self, env, goal_size=0):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space['observation'].shape).prod() + np.prod(env.single_action_space.shape)
            + goal_size,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, goal_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space['observation'].shape).prod() + goal_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) 

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SACTrainer:
    def __init__(
            self,
        env_type,
        env_id="spin_rl", 
        num_envs=4, 
        seed=1, 
        num_steps=128, 
        exp_name=os.path.basename(__file__)[: -len(".py")],
        vf_coef= 0.5,
        gamma=0.99,
        max_grad_norm=0.5,
        policy_lr=3e-4,
        learning_starts= 5e3,
        q_lr= 1e-3,
        policy_frequency=2,
        target_network_frequency=1,
        alpha= 0.2,
        autotune= True,
        batch_size=256,
        buffer_size=1e6,
        tau=0.005,
        goal_size=0
    ):
        self.seed = seed
        self.env_id = env_id
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.gamma = gamma

        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.env_type = env_type
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_type, self.env_id, i, True, self.run_name) for i in range(self.num_envs)],
            autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
        )

        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.autotune = autotune
        self.alpha = alpha
        self.buffer_size = buffer_size
        
        self.envs.single_observation_space.dtype = np.float32

        self.goal_size = goal_size
        self.rb = ReplayBuffer(
            int(self.buffer_size),
            spaces.Box(
                low=float('-inf'), high=float('inf'), shape=(
                    np.array(self.envs.single_observation_space['observation'].shape).prod() 
                    + self.goal_size,
                ), dtype=np.float32

            ),
            self.envs.single_action_space,
            self.device,
            n_envs=self.num_envs,
            handle_timeout_termination=False,
        )
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau


    def train(self, total_timesteps=5000000, save_model=True):
        actor = Actor(self.envs, goal_size=self.goal_size).to(self.device)
        qf1 = SoftQNetwork(self.envs, goal_size=self.goal_size).to(self.device)
        qf2 = SoftQNetwork(self.envs, goal_size=self.goal_size).to(self.device)
        qf1_target = SoftQNetwork(self.envs, goal_size=self.goal_size).to(self.device)
        qf2_target = SoftQNetwork(self.envs, goal_size=self.goal_size).to(self.device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=self.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=self.policy_lr)
        if self.autotune:
            target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=self.q_lr)
        else:
            alpha = self.alpha

        start_time = time.time()

        obs, _ = self.envs.reset(seed=self.seed)
        for global_step in range(total_timesteps):
            if global_step < self.learning_starts:
                actions = np.array(
                    [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                obs_for_action = transform_obs(obs, self.goal_size)
                actions, _, _ = actor.get_action(torch.Tensor(obs_for_action).to(self.device))
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    print(f"global_step={global_step}, episodic_return={reward}")
                    self.writer.add_scalar("charts/episodic_return", reward, global_step)

                for length in infos["final_info"]["episode"]["l"]:
                    self.writer.add_scalar("charts/episodic_length", length, global_step)

            real_next_obs = next_obs.copy()
            for idx, is_done in enumerate(np.logical_or(terminations, truncations)):
                if is_done:
                    real_next_obs[idx] = infos["final_obs"][idx]
            
            obs_to_add = transform_obs(obs, self.goal_size)
            next_obs_to_add = transform_obs(real_next_obs, self.goal_size)
            
            self.rb.add(obs_to_add, next_obs_to_add, actions, rewards, terminations, infos)

            obs = next_obs

            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(torch.tensor(data.next_observations, dtype=torch.float32))
                    qf1_next_target = qf1_target(torch.tensor(data.next_observations, dtype=torch.float32), next_state_actions)
                    qf2_next_target = qf2_target(torch.tensor(data.next_observations, dtype=torch.float32), next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(torch.tensor(data.observations, dtype=torch.float32), torch.tensor(data.actions, dtype=torch.float32)).view(-1)
                qf2_a_values = qf2(torch.tensor(data.observations, dtype=torch.float32), torch.tensor(data.actions, dtype=torch.float32)).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % self.policy_frequency == 0: 
                    for _ in range(
                        self.policy_frequency
                    ):  
                        pi, log_pi, _ = actor.get_action(torch.tensor(data.observations, dtype=torch.float32))
                        qf1_pi = qf1(torch.tensor(data.observations, dtype=torch.float32), pi)
                        qf2_pi = qf2(torch.tensor(data.observations, dtype=torch.float32), pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(torch.tensor(data.observations, dtype=torch.float32))
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                if global_step % self.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    self.writer.add_scalar("losses/alpha", alpha, global_step)
                    if self.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if save_model:
            actor_model_path = f"runs/{self.run_name}/actor.pt"
            critic_model_path = f"runs/{self.run_name}/critic.pt"
            torch.save(actor.state_dict(), actor_model_path)
            torch.save(qf1.state_dict(), critic_model_path)

            evaluate(
                self.env_type,
                actor_model_path,
                make_env,
                self.env_id,
                eval_episodes=10,
                run_name=self.run_name,
                Model=Actor,
                device=self.device,
                gamma=self.gamma,
            )

        self.envs.close()
        self.writer.close()





if __name__ == "__main__":
    pass
