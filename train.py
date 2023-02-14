import gym_driving
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_results(log_folder, title="training phase"):

    def moving_average(values, window): 
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")

    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title(title)
    plt.savefig("logs/fig_after_training.png")

hyperparameters = {
    "learning_rate": 3e-5,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 20,
    "gamma": 0.99,
    "gae_lambda": 0.9,
    "clip_range": 0.4,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "sde_sample_freq": 4,
    "use_sde": True,
    "policy_kwargs": dict(log_std_init=-2,
                       ortho_init=False,
                       activation_fn=nn.ReLU,
                       net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                       ),
}

if __name__ == "__main__":
    log_dir = "logs/"
    callbacks = []

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format("0"))
        torch.cuda.set_device(device)

    env = make_vec_env("Driving-v0", n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env("Driving-v0", vec_env_cls=SubprocVecEnv)

    callbacks.append(EvalCallback(eval_env, best_model_save_path=log_dir, eval_freq=50000, deterministic=True, render=False))
    callbacks.append(CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix="rl_model"))

    model = PPO("MlpPolicy", env, **hyperparameters)

    model.learn(3e6, callback=callbacks)
    model.save("logs/final_model")

    plot_results(log_dir)
