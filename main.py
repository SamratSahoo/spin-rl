from environment import WrappedPenSpinEnv

if __name__ == "__main__":
    env = WrappedPenSpinEnv(render_mode="human")
    current_state, _ = env.reset()
    current_step = 1
    done = False
    while not done:
        action = env.sample()
        next_state, reward, done, _, _ = env.step(action)
        print(f"Step {current_step}: Reward: {reward}")
        current_state = next_state
        current_step += 1
    env.close()
