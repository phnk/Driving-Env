'''
Runs driving-v0 OpenAI gym environment. 
'''

import gym 
import gym_driving
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


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


def train(env_name):
    total_steps = 2400
    num_test = 20
    model_name = 'saved_model'
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    model = A2C(MlpPolicy, env, verbose=0)



    #rewards = []
    #for _ in range(num_test):
    model.learn(total_timesteps=total_steps)
    #    rewards.append(test(env, model))
    #print(rewards)
    #model.load(model_name)
    print(test(env, model))
    model.save(model_name)
    env.close()
    return model


def test(env, model): 
    num_episodes = 10
    rewards = 0
    ob = env.reset()

    done = False
    for ep in range(num_episodes): 
        while not done:
            action, _states = model.predict(ob)
            ob, step_rew, done, _ = env.step(action)
            rewards += step_rew
        ob = env.reset()
    return float(rewards / num_episodes)

if __name__ == '__main__': 
    env = 'Driving-v0'
    train(env)

