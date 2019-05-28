import gym 
import gym_driving
import tensorflow as tf
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO1
import matplotlib.pyplot as plt
import pickle

def graph_reward(ep_reward):
    '''
    Graph info of ep_reward
    '''
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(ep_reward)
    ran = (max(ep_reward) - min(ep_reward)) * 0.1
    ax.set_ylim((min(ep_reward) - ran, max(ep_reward) + ran))
    ax.set_title('Reward per timestep.')
    plt.show()

def main(): 
    ''' 
    Trains baselines PPO.
    '''
    total_steps = (1000 * 1000)
    policy_kwarg = dict(act_fun=tf.nn.tanh, net_arch=[64, 64])
    seed = 1

    env = gym.make('Driving-v0')
    env.seed(seed)
    tf.random.set_random_seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    set_global_seeds(seed)
    

    env = DummyVecEnv([lambda: env])
    model = PPO1('MlpPolicy', env, policy_kwargs=policy_kwarg,
        timesteps_per_actorbatch=10000, optim_batchsize=2500,
        optim_stepsize=3e-4, optim_epochs=6)

    logger = {'rewards': [], 'lengths': []}
    model.learn(total_timesteps=total_steps, gerard_logger=logger, seed=seed)
    model.save('saved_ppo_v1')

    with open('ppo_reward', 'wb') as fp: 
        dic = {'v0': logger}
        pickle.dump(dic, fp)

if __name__ == '__main__': 
    main()
