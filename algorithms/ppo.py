# Implementation Based on Clean RL: https://github.com/vwxyzjn/cleanrl
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sys import platform
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from algorithms.evaluate_agent import evaluate

os.environ["MUJOCO_GL"] = "glfw" if platform == "darwin" else "osmesa"


def make_env(env_type, env_id, idx, capture_video, run_name, is_eval=False):
    def thunk():
        if capture_video and idx == 0:
            env = env_type(
                reward_type="dense", temp_dir=f"./tmp/{env_id}_{idx}", 
                max_episode_steps=300
            )
            if not is_eval:
                env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
            else:
                env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/eval")

            env.recorded_frames = []
        else:
            env = env_type(
                reward_type="dense", temp_dir=f"./tmp/{env_id}_{idx}", 
                max_episode_steps=300
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['observation'].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['observation'].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PPOTrainer:
    def __init__(
    self, 
    env_type,
    env_id="spin_rl", 
    num_envs=4, 
    seed=1, 
    num_steps=128, 
    num_minibatches=4,
    exp_name=os.path.basename(__file__)[: -len(".py")],
    learning_rate=2.5e-4,
    anneal_lr=True,
    target_kl=None,
    gamma = 0.99,
    gae_lambda = 0.95,
    update_epochs=4,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef= 0.0,
    vf_coef= 0.5,
    max_grad_norm=0.5
    ):
        self.seed = seed
        self.env_id = env_id
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.exp_name = exp_name
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_type, self.env_id, i, True, self.run_name) for i in range(self.num_envs)],
            autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
        )

        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef =ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm


    def train(self, total_timesteps=20000000, save_model=True):
        self.num_iterations = total_timesteps // self.batch_size
        agent = Agent(self.envs).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5)

        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space['observation'].shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        global_step = 0
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs['observation']).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs['observation']).to(self.device), torch.Tensor(next_done).to(self.device)

            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    print(f"global_step={global_step}, episodic_return={reward}")
                    self.writer.add_scalar("charts/episodic_return", reward, global_step)

                for length in infos["final_info"]["episode"]["l"]:
                    self.writer.add_scalar("charts/episodic_length", length, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space['observation'].shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            evaluate(
                model_path,
                make_env,
                self.env_id,
                eval_episodes=10,
                run_name=self.run_name,
                Model=Agent,
                device=self.device,
                gamma=self.gamma,
            )

        self.envs.close()
        self.writer.close()

