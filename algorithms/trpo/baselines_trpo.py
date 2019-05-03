import gym 
import gym_driving
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
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
    Trains baselines TRPO.
    '''
    total_steps = 1200 * 400
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32])

    env = gym.make('Driving-v0')
    env = DummyVecEnv([lambda: env])
    model = TRPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, 
        timesteps_per_batch=12000)

    logger = []
    model.learn(total_timesteps=total_steps, gerard_logger=logger)
    model.save('saved_trpo')

    with open('trpo_rew_per_episode', 'wb') as fp: 
        pickle.dump(logger, fp)

    
    avg_rew = []
    avg = 10
    for bottom_ind in range(0, len(logger), avg):
        avg_rew.append(sum(logger[bottom_ind: bottom_ind + avg]) / avg)

    print(avg_rew)
    graph_reward(avg_rew)

    
    ob = env.reset()
    while True: 
        action, _states = model.predict(ob)
        ob, rew, done, _ = env.step(action)
        if done: 
            break
        env.render()
    env.close()

def render(): 
    env = gym.make('Driving-v0')
    env = DummyVecEnv([lambda: env])
    model = TRPO.load('saved_trpo')

    with open('trpo_rew_per_episode', 'rb') as fp: 
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
    render()
