from typing import Callable

import gymnasium as gym
import torch


def evaluate(
        env_type,
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_type, env_id, 0, capture_video, run_name, gamma, is_eval=True)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for reward in infos["final_info"]["episode"]["r"]:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={reward}")
                episodic_returns.append(reward)
                break
        obs = next_obs

    return episodic_returns