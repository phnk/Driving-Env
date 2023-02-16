import time
import gym
import gym_driving

from stable_baselines3 import PPO

def main():
    env = gym.make('Driving-v0')
    model = PPO("MlpPolicy", env)
    model = model.load("logs/final_model")

    i = 0 

    env.reset()
    while True:
        if i % 2 == 0:
            # random action
            action = env.action_space.sample()
        else:
            # trained agent
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

            i += 1

        # matching the update rate of the physics engine, remove if you want it to go fast
        time.sleep(1/120)

    env.close()


if __name__ == '__main__': 
    main()

