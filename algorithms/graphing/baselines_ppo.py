import gym 
import gym_driving
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
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
    Trains baselines PPO2.
    '''
    total_steps = (1000 * 800)
    policy_kwarg = dict(act_fun=tf.nn.relu, net_arch=[64, 64])

    env = gym.make('Driving-v0')
    env.seed(1)
    env = DummyVecEnv([lambda: env])
    model = PPO2('MlpPolicy', env, policy_kwargs=policy_kwarg, verbose=1)

    logger = {'rewards': [], 'lengths': []}
    model.learn(total_timesteps=total_steps)
    model.save('saved_ppo_v1')

    with open('ppo_v1_rew', 'wb') as fp: 
        dic = {'ppo': logger}
        pickle.dump(dic, fp)

def render(): 
    env = gym.make('Driving-v0')
    env.seed(3)
    env = DummyVecEnv([lambda: env])
    model = PPO1.load('saved_ppo_v0')

    with open('ppo_v1_rew', 'rb') as fp: 
        logger = pickle.load(fp)

    reward = [] 
    ob = env.reset()
    while True: 
        action, _states = model.predict(ob)
        ob, rew, done, _ = env.step(action)
        print(round(rew.item(), 3))
        if done: 
            break
        reward.append(rew.item())
        env.render()
    env.close()

    graph_reward(reward)

    print('SUM OF REWARD: ', sum(reward))
    for ind in range(len(reward) - 2, -1, -1): 
        reward[ind] += reward[ind + 1] * 0.99
    print('SUM OF REWARD GAMMA 0.99: ', sum(reward))

if __name__ == '__main__': 
    main()
