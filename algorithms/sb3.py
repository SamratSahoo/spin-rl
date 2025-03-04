import time
from stable_baselines3 import PPO, SAC, TD3, DDPG
import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
import os
from stable_baselines3.common.noise import (
    OrnsteinUhlenbeckActionNoise,
    NormalActionNoise,
)


class SetBestModelCallback(BaseCallback):
    parent: EvalCallback

    def __init__(self, sb3, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.sb3 = sb3

    def _on_step(self) -> bool:
        assert self.parent is not None

        self.sb3.set_best_model()
        model_recording = f"{self.sb3.eval_recording_folder}/{self.sb3.algorithm}_{round(time.time() * 1000)}.mp4"
        model_path = f"{os.path.abspath(self.sb3.best_model_save_folder)}/{self.sb3.algorithm}_best_model"

        self.sb3.run_evaluation(model_path=model_path, recording=model_recording)

        return True


class StableBaseLines3Runner:
    def __init__(
        self,
        env,
        eval_env,
        algorithm="sac",
        checkpoint=None,
        best_model_save_folder=None,
        best_model_name=None,
        eval_recording_folder=None,
        eval_freq=500,
    ):
        self.env = env
        self.algorithm = algorithm
        self.eval_env = eval_env
        self.model = None
        self.best_model = None
        self.checkpoint = checkpoint
        self.model_class = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}[algorithm]
        self.best_model_name = best_model_name
        self.best_model_save_folder = best_model_save_folder
        self.eval_recording_folder = eval_recording_folder

        self.set_best_model_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=best_model_save_folder,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=SetBestModelCallback(sb3=self, verbose=1),
        )

    def train(
        self, timesteps=10000, device="cpu", policy="MlpPolicy", model_settings=dict()
    ):
        if model_settings.get("action_noise_type", "") == "OrnsteinUhlenbeck":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(self.env.action_space.shape[-1]),
                sigma=model_settings["action_noise_sigma"]
                * np.ones(self.env.action_space.shape[-1]),
            )
            del model_settings["action_noise_type"]
            del model_settings["action_noise_sigma"]
            model_settings["action_noise"] = action_noise
        elif model_settings.get("action_noise_type", "") == "Normal":
            action_noise = NormalActionNoise(
                mean=np.zeros(self.env.action_space.shape[-1]),
                sigma=model_settings["action_noise_sigma"]
                * np.ones(self.env.action_space.shape[-1]),
            )
            del model_settings["action_noise_type"]
            del model_settings["action_noise_sigma"]
            model_settings["action_noise"] = action_noise

        if self.checkpoint and os.path.exists(f"{self.checkpoint}.zip"):
            self.model = self.model_class.load(
                f"{os.getcwd()}/{self.checkpoint}", env=self.env, device=device
            )
        else:
            self.model = self.model_class(
                policy, self.env, verbose=1, device=device, **model_settings
            )

        self.model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            callback=[self.set_best_model_callback],
        )

    def set_best_model(self):
        os.rename(
            f"{self.best_model_save_folder}/best_model.zip",
            f"{self.best_model_save_folder}/{self.best_model_name}",
        )

    def run_evaluation(self, model_path, recording=None):

        current_state, _ = self.eval_env.reset()

        if recording:
            self.eval_env.start_recording()

        self.best_model = self.model_class.load(model_path)

        while True:
            action, _ = self.best_model.predict(current_state, deterministic=True)
            next_state, reward, done, _, info = self.eval_env.step(action)
            if done:
                break
            current_state = next_state

        if recording:
            self.eval_env.stop_recording(recording_name=recording)
