import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from algorithms.evaluate_agent import evaluate
from algorithms.utils import make_env

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space['observation'].shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space['observation'].shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class DDPGTrainer:
    def __init__(
    self, 
    env_type,
    env_id="spin_rl", 
    seed=1, 
    exp_name=os.path.basename(__file__)[: -len(".py")],
    learning_rate=2.5e-4,
    learning_starts=25e3,
    exploration_noise=0.1,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_frequency=2,
    buffer_size=1e6
    ):
        self.seed = seed
        self.exp_name = exp_name
        self.env_id = env_id
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_type, self.env_id, 0, True, self.run_name)],
            autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
        )

        self.writer = SummaryWriter(f"runs/{self.run_name}")

        self.learning_rate = learning_rate
        self.env_type = env_type

        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = QNetwork(self.envs).to(self.device)
        self.qf1_target = QNetwork(self.envs).to(self.device)
        self.target_actor = Actor(self.envs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)
        self.buffer_size = buffer_size
        self.envs.single_observation_space['observation'].dtype = np.float32

        self.rb = ReplayBuffer(
            int(self.buffer_size),
            self.envs.single_observation_space['observation'],
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts

        self.batch_size = batch_size
        self.gamma = gamma

        self.tau = tau
        self.policy_frequency = policy_frequency
    
    def train(self, timesteps=5000000, save_model=True):
    
        obs, _ = self.envs.reset(seed=self.seed)
        for global_step in range(timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array([self.envs.single_action_space.sample()])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs['observation']).to(self.device))
                    actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    print(f"global_step={global_step}, episodic_return={reward}")
                    self.writer.add_scalar("charts/episodic_return", reward, global_step)

                for length in infos["final_info"]["episode"]["l"]:
                    self.writer.add_scalar("charts/episodic_length", length, global_step)


            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, is_done in enumerate(np.logical_or(terminations, truncations)):
                if is_done:
                    real_next_obs[idx] = infos["final_obs"][idx]

            self.rb.add(obs['observation'], real_next_obs['observation'], actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (qf1_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                self.q_optimizer.zero_grad()
                qf1_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        if save_model:
            actor_model_path = f"runs/{self.run_name}/actor.pt"
            critic_model_path = f"runs/{self.run_name}/critic.pt"
            torch.save(self.actor.state_dict(), actor_model_path)
            torch.save(self.qf1.state_dict(), critic_model_path)

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