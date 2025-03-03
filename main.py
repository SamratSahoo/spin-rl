from environment import WrappedPenSpinEnv
import gymnasium as gym

if __name__ == "__main__":
    env = WrappedPenSpinEnv(render_mode="rgb_array", max_episode_steps=300)
    current_state, _ = env.reset()
    current_step = 1
    done = False
    env.start_recording()
    while not done:
        action = env.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        current_state = next_state
        current_step += 1
    env.stop_recording()
    env.close()
