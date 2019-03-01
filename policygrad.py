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
    def __init__(self, in_shape, out_shape): 
        super().__init__()

        self.hidden_act = nn.Tanh()

        self.layers = nn.ModuleList([
            nn.Linear(in_shape, 128),
            self.hidden_act, 
            nn.Linear(128, 128), 
            self.hidden_act, 
            nn.Linear(128, out_shape)
        ])

        for layer in self.layers: 
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight,
                    gain=nn.init.calculate_gain('tanh'))

    def forward(self, state): 
        for layer in self.layers: 
            state = layer(state)
        return state


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
    policy = Policy(config['ob_dim'], config['action_dim'])
    optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])
    # Std 
    std = torch.ones(config['action_dim'], requires_grad=True)
    # Recording 
    ep_reward = []
    ep_loss = []
    
    # Try to load from checkpoint
    try: 
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint['policy'])
        std = checkpoint['std']
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        optim.load_state_dict(checkpoint['optim'])
        ep_reward = checkpoint['ep_reward']
        ep_loss = checkpoint['ep_loss']
        print(f'Resuming training from Epoch {len(ep_loss)}')
    except FileNotFoundError:
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        print('NOTE: Training from scratch.')
            
    return (policy, optim, std, ep_reward, ep_loss)


def save_checkpoint(checkpoint_path, policy, optim, std, ep_reward, 
    ep_loss):
    '''
    Saves checkpoint.
    '''
    torch.save({
        'policy': policy.state_dict(), 
        'optim': optim.state_dict(), 
        'std': std, 
        'ep_reward': ep_reward, 
        'ep_loss': ep_loss
    }, checkpoint_path)


def main(): 
    '''
    Training procedure.
    '''

    # Hyperparameters
    config = { 
    'action_dim': 4,
    'ob_dim': 24,
    'max_trajectory': 800,
    'batch_size': 8,
    'discount': 0.90,
    'policy_lr': 1e-3,
    'std_lr': 1e-3,
    'epochs': 40,
    }

    # Load checkpoint
    checkp_path = 'checkpoint.tar'
    policy, optim, std, ep_reward, ep_loss = \
        load_checkpoint(checkp_path, config)

    # Train over epochs (batches of episodes)
    env = gym.make('BipedalWalker-v2')
    for ep in range(1, config['epochs'] + 1):
        ep_reward.append(0)
        ep_loss.append(0)
        won = 0
        # Run episodes for one batch
        for episode in range(config['batch_size']): 
            log_prob = []
            rewards = []
            ob = torch.from_numpy(env.reset()).float()
            # Run single episode
            for step in range(config['max_trajectory']):
                dist = norm_dist(policy(ob), torch.exp(std))
                action = dist.sample()
                log_prob.append(dist.log_prob(action))
                ob, reward, done, _ = env.step(action.numpy())
                ob = torch.from_numpy(ob).float()
                rewards.append(reward)
                if done: 
                    break
            # Accumulate gradients 
            running_rew = 0
            for prob, reward in zip(reversed(log_prob), reversed(rewards)):
                running_rew = reward + config['discount'] * running_rew
                ep_loss[-1] -= prob * reward
            ep_reward[-1] += sum(rewards)
        # Backward        
        optim.zero_grad() 
        ep_loss[-1] /= config['batch_size']
        ep_loss[-1].sum().backward()
        # Recording
        ep_reward[-1] /= config['batch_size']
        ep_loss[-1] = -ep_loss[-1].sum().item()
        # Step 
        optim.step()

        # Save checkpoint 
        if ep % 10 == 0: 
            print(f'Epoch {len(ep_loss)}\tReward:\t{round(ep_reward[-1], 2)}\t'
                f'Loss:\t{round(ep_loss[-1], 2)}')
            save_checkpoint(checkp_path, policy, optim, std,  ep_reward, 
                ep_loss)

    # Graph info 
    ep = [i * config['batch_size'] for i in range(len(ep_loss))]
    plt.plot(ep, ep_loss)
    plt.title('Loss per episode.')
    plt.show()

    policy.eval()
    # Render 
    ob = torch.from_numpy(env.reset()).float()
    for _ in range(config['max_trajectory']): 
        with torch.no_grad():
            env.render()
            dist = torch.distributions.normal.Normal(
                        policy(ob), torch.exp(std))
            action = dist.sample()
            ob, reward, done, _ = env.step(action.numpy())
            ob = torch.from_numpy(ob).float()
            if done: 
                break
    env.close()

if __name__ == '__main__': 
    main()
