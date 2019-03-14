'''
Policy gradient with REINFORCE over episodes.

Standard REINFORCE, z scores rewards over single episodes. Averages
gradient over episodes.
'''

import torch 
import gym 
import gym_driving
import numpy as np
import torch.nn as nn
from util import *


def load_checkpoint(checkpoint_path, config): 
    '''
    Loads a checkpoint if it exists. Otherwise, initializes.
    '''
    # Policy
    policy = Network(config['ob_dim'], config['policy_hidden_units'] + 
        [config['action_dim']])
    optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])
    # Std 
    std = torch.ones(config['action_dim'], requires_grad=True)
    # Recording 
    ep_reward = []
    
    # Try to load from checkpoint
    try: 
        checkpoint = torch.load(checkpoint_path)

        # Reset policy
        policy_param = checkpoint['policy_param']
        policy = Network(config['ob_dim'], policy_param)
        optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])

        # Load state dicts
        policy.load_state_dict(checkpoint['policy'])
        std = checkpoint['std']
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        optim.load_state_dict(checkpoint['optim'])
        ep_reward = checkpoint['ep_reward']
        print(f'Resuming training from episode {len(ep_reward)}')
    except FileNotFoundError:
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        print('NOTE: Training from scratch.')
            
    return (policy, optim, std, ep_reward)


def save_checkpoint(checkpoint_path, policy, optim, std, ep_reward, config):
    '''
    Saves checkpoint.
    '''
    torch.save({
        'policy': policy.state_dict(), 
        'policy_param': config['policy_hidden_units'] + [config['action_dim']],
        'optim': optim.state_dict(), 
        'std': std, 
        'ep_reward': ep_reward
    }, checkpoint_path)


def process_rewards(rewards, discount):
    running_rew = 0
    for ind in range(len(rewards) - 1, -1, -1):
        # Save running reward as estimate for future reward
        running_rew = rewards[ind] + discount * running_rew
        rewards[ind] = running_rew
    # Normalize rewards 
    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()
    return rewards


def main(): 
    '''
    Training procedure.
    '''

    # Hyperparameters
    config = { 
    'action_dim': None,
    'ob_dim': None,
    'policy_hidden_units': [],
    'max_trajectory': 500,
    'episodes': 4,
    'discount': 0.98,
    'policy_lr': 1e-1,
    'std_lr': 1e-2,
    'epochs': 100
    }

    env = gym.make('Driving-v0')
    config['action_dim'] = env.action_space.low.size
    config['ob_dim'] = env.observation_space.low.size

    # Load checkpoint
    checkp_path = 'checkpoint_reinforce.tar'
    policy, optim, std, episode_reward = load_checkpoint(checkp_path, config)
    device = (torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))
    policy.to(device)
    std.to(device)

    # Train over epochs (batches of normalized episodes)
    for ep in range(1, config['epochs'] + 1):
        ep_loss = 0
        # Run fixed number of episodes
        for _ in range(config['episodes']):
            rewards = []
            log_probs = []
            ob = torch.from_numpy(env.reset()).float().to(device)
            # Run single episode
            for _ in range(config['max_trajectory']):
                action, prob = classify_continuous(policy(ob), std)
                ob, reward, done, _ = env.step(action)
                ob = torch.from_numpy(ob).float().to(device)
                rewards.append(reward)
                log_probs.append(prob)
                if done:
                    break
            # Record, normalize rewards 
            episode_reward.append(sum(rewards))
            rewards = process_rewards(rewards, config['discount'])
            # Calculate and average loss over steps in trajectory
            loss = 0
            for prob, rew in zip(log_probs, rewards): 
                loss -= prob * rew
            ep_loss += loss / len(log_probs)

        optim.zero_grad() 
        # Average loss over number of episodes
        ep_loss /= config['episodes']
        # Average over output dimensions and backprop
        ep_loss.mean().backward()
        # Step with optim
        optim.step()

        # Save checkpoint 
        if ep % 5 == 0: 
            print('Sigma: ', [round(i, 3) for i in std.detach().numpy()])
            print(f'Episode {len(episode_reward)}\t'
                'Reward:\t', round(sum(episode_reward[-config['episodes']:]) /
                config['episodes'], 2))
            save_checkpoint(checkp_path, policy, optim, std,  episode_reward, 
                config)
                
    graph_reward(episode_reward)

    policy.eval()
    # Render 
    ob = torch.from_numpy(env.reset()).float().to(device)
    for _ in range(config['max_trajectory']): 
        with torch.no_grad():
            env.render()
            action, _ = classify_continuous(policy(ob), std)
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float().to(device)
            if done: 
                break
    env.close()

if __name__ == '__main__': 
    main()
