import gymnasium as gym

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