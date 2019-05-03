import gym 
import gym_driving
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
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
    ax.set_title('Reward per episode.')
    plt.show()

def main(): 
    ''' 
    Trains baselines PPO.
    '''
    total_steps = 1200 * 800
    policy_kwarg = dict(act_fun=tf.nn.relu, net_arch=[64, 64])

    env = gym.make('Driving-v0')
    env.seed(1)
    env = DummyVecEnv([lambda: env])
    model = PPO1('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwarg,
    timesteps_per_actorbatch=4096)

    logger = {'rewards': [], 'lengths': []}
    model.learn(total_timesteps=total_steps, gerard_logger=logger)
    model.save('saved_ppo_v1')

    with open('ppo_v1_rew', 'wb') as fp: 
        dic = {'ppo': logger}
        pickle.dump(dic, fp)

def render(): 
    env = gym.make('Driving-v1')
    env = DummyVecEnv([lambda: env])
    model = TRPO.load('saved_ppo_v1')

    with open('ppo_rew_per_episode', 'rb') as fp: 
        reward = pickle.load(fp)
    print(reward)
    ob = env.reset()
    while True: 
        action, _states = model.predict(ob)
        ob, rew, done, _ = env.step(action)
        if done: 
            break
        env.render()
    env.close()

if __name__ == '__main__': 
    main()
