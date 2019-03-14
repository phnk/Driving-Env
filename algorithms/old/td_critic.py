'''
Policy gradient with REINFORCE over steps.

Z scores rewards over potentially multiple episodes, averages gradient
over batches of steps.
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
    # Critic 
    critic = Network(config['ob_dim'], config['critic_hidden_units'] + [1])
    critic_optim = torch.optim.Adam(critic.parameters(), config['critic_lr'])
    # Std 
    std = torch.ones(config['action_dim'], requires_grad=True)
    # Recording 
    ep_reward = []
    
    # Try to load from checkpoint
    try: 
        checkpoint = torch.load(checkpoint_path)

        # Reset policy
        policy_param = checkpoint['policy_param']
        policy = Network(*policy_param)
        optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])
        # Reset critic
        critic_param = checkpoint['critic_param']
        critic = Network(*critic_param)
        critic_optim = torch.optim.Adam(critic.parameters(), 
            config['critic_lr'])
        # Load state dicts
        policy.load_state_dict(checkpoint['policy'])
        critic.load_state_dict(checkpoint['critic'])
        std = checkpoint['std']
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        optim.load_state_dict(checkpoint['optim'])
        critic_optim.load_state_dict(checkpoint['critic_optim'])
        ep_reward = checkpoint['ep_reward']
        print(f'Resuming training from episode {len(ep_reward)}')
    except FileNotFoundError:
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        print('NOTE: Training from scratch.')
            
    return (policy, optim, critic, critic_optim, std, ep_reward)


def save_checkpoint(checkpoint_path, policy, optim, critic, critic_optim, 
    std, ep_reward, config):
    '''
    Saves checkpoint.
    '''
    torch.save({
        'policy': policy.state_dict(), 
        'policy_param': (config['ob_dim'], config['policy_hidden_units'] + 
            [config['action_dim']]),
        'critic': critic.state_dict(), 
        'critic_param': (config['ob_dim'], config['critic_hidden_units'] + [1]),
        'optim': optim.state_dict(), 
        'critic_optim': critic_optim.state_dict(),
        'std': std, 
        'ep_reward': ep_reward
    }, checkpoint_path)


def main(): 
    '''
    Calls training procedure.
    '''

    # Hyperparameters
    config = { 
    'action_dim': None,
    'ob_dim': None,
    'policy_hidden_units': [],
    'critic_hidden_units': [],
    'policy_lr': 1e-1,
    'critic_lr': 1e-2,
    'std_lr': 1e-2,
    'max_steps': 500,
    'batch_size': 2000,
    'max_episodes': 4, 
    'discount': 0.98,
    'epochs': 100,
    }

    env = gym.make('CartPole-v1')
    config['action_dim'] = 2 #env.action_space.low.size
    config['ob_dim'] = env.observation_space.low.size

    # Load checkpoint
    checkp_path = 'checkpoint_ac.tar'
    policy, optim, critic, critic_optim, std, episode_reward = \
        load_checkpoint(checkp_path, config)
    device = (torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))
    policy.to(device)
    critic.to(device)
    std.to(device)

    critic_loss = nn.MSELoss(reduction='mean')

    # Train over epochs (batches of normalized steps)
    for ep in range(1, config['epochs'] + 1):
        # Recording for batch
        log_probs = []
        td_error = []
        # Recording for episodes within a batch
        episode_start_step = 0
        episodes = 1
        rewards = []
        critic_rewards = []

        # Run fixed number of steps with multiple episodes
        step = 0
        ob = torch.from_numpy(env.reset()).float().to(device)
        while step < config['batch_size']:
            # Record critic reward based off observation
            critic_rewards.append(critic(ob))
            # Perform action on env, recover new ob, reward, done
            action, prob = classify_discrete(policy(ob), std)
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float().to(device)
            
            # Append true reward from step and log prob 
            rewards.append(reward)
            log_probs.append(prob)
            step += 1

            # If done with episode
            if done or (step - episode_start_step >= config['max_steps']):
                episodes += 1
                episode_reward.append(sum(rewards))
                # Calculate td targets
                critic_rewards = torch.stack(critic_rewards)
                td_targets = config['discount'] * torch.cat(
                    (critic_rewards[1:].view(-1).data, torch.tensor([0.]))) + \
                    torch.tensor(rewards)
                # Calculate td error
                td_error.extend((td_targets - critic_rewards.view(-1)).tolist())
                # Compute critic loss
                critic_optim.zero_grad()
                critic_loss(critic_rewards, td_targets.view(-1, 1)).backward()
                critic_optim.step()

                # Exit if can't run another episode in batch size
                if step + config['max_steps'] > config['batch_size']:
                    break
                # Exit if over number of episodes desired
                if episodes > config['max_episodes']:
                    break
                # Continue otherwise
                episode_start_step = step 
                ob = torch.from_numpy(env.reset()).float().to(device)
                rewards = []
                critic_rewards = []

        td_error = z_score_rewards(td_error)
        # Compute actor loss
        loss = 0
        for prob, error in zip(log_probs, td_error): 
            loss -= prob * error
        loss /= step
        optim.zero_grad() 
        loss.mean().backward()
        optim.step()

        # Save checkpoint 
        if ep % 5 == 0: 
            print('Sigma: ', [round(i, 3) for i in std.detach().numpy()])
            print(f'Episode {len(episode_reward)}\t'
                  'Reward:\t', round(sum(episode_reward[-30:])
                  / 30, 2))
            save_checkpoint(checkp_path, policy, optim, critic, critic_optim, 
                std,  episode_reward, config)

    # Graph info 
    graph_reward(episode_reward)

    # Render 
    ob = torch.from_numpy(env.reset()).float().to(device)
    for step in range(config['max_steps']): 
        with torch.no_grad():
            env.render()
            action, prob = classify_discrete(policy(ob), std)
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float().to(device)
            if done: 
                break
    env.close()

if __name__ == '__main__': 
    main()
