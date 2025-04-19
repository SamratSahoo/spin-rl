import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torch.distributions import Normal
from algorithms.utils import make_env
import gymnasium as gym
import time
import os
import random

from algorithms.evaluate_agent import evaluate
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, env, goal_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space['observation'].shape).prod() + goal_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_sigma = nn.Linear(256, np.prod(env.single_action_space.shape))


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = F.relu(self.fc_sigma(x)) + 1e-6
        return mu, sigma

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mu, sigma = self(state)
        distribution = Normal(mu, sigma)

        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob

class VPGTrainer:
    def __init__(self, env_type, exp_name=os.path.basename(__file__)[: -len(".py")],env_id='spin_rl', seed=1, gamma=0.99, 
                 use_baseline=True, 
                 learning_rate=1e-2, goal_size=0):
        self.seed = seed
        self.env_id = env_id
        self.exp_name = exp_name
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        self.env_id = env_id
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_type, self.env_id, 0, True, self.run_name)],
            autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
        )
        self.envs.single_observation_space['observation'].dtype = np.float32
        self.goal_size = goal_size
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_baseline = use_baseline
        self.actor = Actor(self.envs, goal_size=self.goal_size).to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)
        self.writer = SummaryWriter(f"runs/{self.run_name}")

        self.env_type = env_type


    def train(self, total_timesteps=5000000, save_model=True):
        obs, _ = self.envs.reset(seed=self.seed)
        log_probs = []
        rewards = []
        for global_step in range(total_timesteps):
            if self.goal_size > 0:
                obs_for_action = np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)
            else:
                obs_for_action = obs['observation']
            action, probs = self.actor.get_action(torch.tensor(obs_for_action, dtype=torch.float32, device=self.device))
            action = action.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

            next_obs, reward, terminations, truncations, infos = self.envs.step(action)

            if "final_info" in infos:
                for ep_reward in infos["final_info"]["episode"]["r"]:
                    print(f"global_step={global_step}, episodic_return={ep_reward}")
                    self.writer.add_scalar("charts/episodic_return", ep_reward, global_step)

                for length in infos["final_info"]["episode"]["l"]:
                    self.writer.add_scalar("charts/episodic_length", length, global_step)

            log_probs.append(probs)
            rewards.append(sum(reward))

            obs = next_obs
            
            real_next_obs = next_obs.copy()
            for idx, is_done in enumerate(np.logical_or(terminations, truncations)):
                if is_done:
                    real_next_obs[idx] = infos["final_obs"][idx]
                    log_probs = torch.stack(log_probs,dim = 0).to(self.device).squeeze(-1)
                    rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
                    rewards = rewards.to(self.device).squeeze(-1)
                    discounted_r = torch.zeros_like(rewards)
                    running_add = 0

                    for t in reversed(range(0,rewards.size(-1))):
                        running_add = running_add*self.gamma + rewards[t]
                        discounted_r[t] = running_add

                    G = discounted_r

                    if self.use_baseline:
                        G = (G - G.mean())/(G.std() + 1e-9)

                    loss = -(log_probs * G.detach()).mean()
                    self.writer.add_scalar("losses/actor_loss", loss, global_step)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    log_probs = []
                    rewards = []

        if save_model:
            actor_model_path = f"runs/{self.run_name}/actor.pt"
            torch.save(self.actor.state_dict(), actor_model_path)

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
        self.writer.close()
        self.envs.close()
