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
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal as norm_dist


class Policy(nn.Module): 
    '''
    Basic NN policy.
    '''
    def __init__(self, in_shape, layer_units):
        '''
        layer_unit is list of int, output of each layer.
        len(layer_unit) must be >= 1
        '''
        super().__init__()
        self.hidden_act = nn.Tanh()

        layers = []
        for out_shape in layer_units[:-1]: 
            layers.append(nn.Linear(in_shape, out_shape))
            layers.append(self.hidden_act)
            in_shape = out_shape
        layers.append(nn.Linear(in_shape, layer_units[-1]))

        self.layers = nn.ModuleList(layers)

        for layer in self.layers: 
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight,
                    gain=nn.init.calculate_gain('tanh'))

    def forward(self, state): 
        for layer in self.layers: 
            state = layer(state)
        return state


def classify_continuous(means, stds): 
    dist = norm_dist(means, torch.exp(stds))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.cpu().numpy(), log_prob


def classify_discrete(logits): 
    probs = nn.functional.softmax(logits, dim=0)
    action = torch.multinomial(probs, 1).cpu().item()
    log_prob = probs[action].log()
    return action, log_prob


class SGDOptim: 
    '''
    Wrapper for SGD optimizer.
    '''
    def __init__(self, model, optim, scheduler=None, clip=None): 
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.clip = clip

    def step(self, val_loss=None): 
        if self.scheduler is not None: 
            self.scheduler.step(val_loss)
        if self.clip is not None: 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                self.clip)
        self.optim.step()
    
    def state_dict(self): 
        return self.optim.state_dict(), self.scheduler.state_dict()

    def load_state_dict(self, state): 
        self.optim.load_state_dict(state[0])
        self.scheduler.load_state_dict(state[1])

    def add_param_group(self, *args): 
        optim.add_param_group(*args)

    def zero_grad(self): 
        self.optim.zero_grad()


def load_checkpoint(checkpoint_path, config): 
    '''
    Loads a checkpoint if it exists. Otherwise, initializes.
    '''
    # Policy
    policy = Policy(config['ob_dim'], config['policy_hidden_units'] + 
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
        policy = Policy(config['ob_dim'], policy_param)
        optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])

        # Load state dicts
        policy.load_state_dict(checkpoint['policy'])
        std = checkpoint['std']
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        optim.load_state_dict(checkpoint['optim'])
        ep_reward = checkpoint['ep_reward']
        print(f'Resuming training from Epoch {len(ep_reward)}')
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


def graph_reward(ep_reward):
    '''
    Graph info.
    '''
    ep = [i for i in range(len(ep_reward))]
    plt.plot(ep, ep_reward)
    ran = (max(ep_reward) - min(ep_reward)) * 0.1
    plt.ylim((min(ep_reward) - ran, max(ep_reward) + ran))
    plt.title('Reward per episode.')
    plt.show()


def main(): 
    '''
    Training procedure.
    '''

    # Hyperparameters
    config = { 
    'action_dim': None,
    'ob_dim': None,
    'policy_hidden_units': [20],
    'max_trajectory': 300,
    'episodes': 5,
    'discount': 0.90,
    'policy_lr': 1e-3,
    'std_lr': 1e-2,
    'epochs': 100
    }

    env = gym.make('Driving-v0')
    config['action_dim'] = env.action_space.low.size
    config['ob_dim'] = env.observation_space.low.size

    # Load checkpoint
    checkp_path = 'checkpoint.tar'
    policy, optim, std, ep_reward = load_checkpoint(checkp_path, config)
    device = (torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))
    policy.to(device)
    std.to(device)

    # Train over epochs (batches of normalized episodes)
    for ep in range(1, config['epochs'] + 1):
        ep_reward.append(0)
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
            ep_reward[-1] += sum(rewards)
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

        # Recording
        ep_reward[-1] /= config['episodes']

        # Save checkpoint 
        if ep % 5 == 0: 
            print(f'Epoch {ep}\tReward:\t{round(ep_reward[-1], 2)}')
            save_checkpoint(checkp_path, policy, optim, std,  ep_reward, config)
                

    # graph_reward(ep_reward)

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
