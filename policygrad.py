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

        self.hidden_act = nn.ReLU()
        self.std = nn.Parameter(torch.ones(out_shape))

        self.layers = nn.ModuleList([
            nn.Linear(in_shape, 60),
            self.hidden_act, 
            nn.Linear(60, 24), 
            self.hidden_act, 
            nn.Linear(24, out_shape)
        ])

        for layer in self.layers: 
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, state): 
        for layer in self.layers: 
            state = layer(state)
        return state


def load_checkpoint(checkpoint_path, config): 
    model = Policy(config['ob_dim'], config['action_dim'])
    optim = torch.optim.Adam(model.parameters(), lr=config['alpha'])
    ep_reward = []
    ep_loss = []
    try: 
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        ep_reward = checkpoint['ep_reward']
        ep_loss = checkpoint['ep_loss']
    except FileNotFoundError:
        print('NOTE: Training from scratch.')
    return model, optim, ep_reward, ep_loss


def save_checkpoint(checkpoint_path, model, optim, ep_reward, ep_loss):
    torch.save({
        'model': model.state_dict(), 
        'optim': optim.state_dict(), 
        'ep_reward': ep_reward, 
        'ep_loss': ep_loss
    }, checkpoint_path)


def main(): 
    '''
    Training procedure.
    '''

    # Hyperparameters
    config = { 
    'action_dim': 1,
    'ob_dim': 2,
    'max_trajectory': 1000,
    'batch_size': 48,
    'discount': 0.90,
    'alpha': 0.01,
    'epochs': 15
    }

    checkpoint_path = 'checkpoint.tar'
    model, optim, ep_reward, ep_loss = load_checkpoint(checkpoint_path, config)

    # Train over epochs (batches of episodes)
    env = gym.make('MountainCarContinuous-v0')
    for _ in range(config['epochs']):
        ep_reward.append(0)
        ep_loss.append(0)
        # Run episodes for one batch
        for episode in range(config['batch_size']): 
            log_prob = []
            rewards = []
            ob = torch.from_numpy(env.reset()).float()
            # Run single episode
            for step in range(config['max_trajectory']):
                dist = norm_dist(model(ob), torch.exp(model.std))
                action = dist.sample()
                log_prob.append(dist.log_prob(action))

                ob, reward, done, _ = env.step(action)
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
        ep_loss[-1].backward()
        optim.step()
        # Print reward and loss over the episodes of the batch
        ep_reward[-1] /= config['batch_size']
        ep_loss[-1] = -ep_loss[-1].item()
        print(f'Epoch {len(ep_loss)}\tReward:\t{round(ep_reward[-1], 2)}\t'
            f'Loss:\t{round(ep_loss[-1], 2)}')

    # Save checkpoint 
    save_checkpoint(checkpoint_path, model, optim, ep_reward, ep_loss)  
    # Print info 
    ep = [i * 48 for i in range(len(ep_loss))]
    plt.plot(ep, ep_reward)
    plt.title('Reward per episode.')
    plt.show()

    model.eval()
    # Render 
    ob = torch.from_numpy(env.reset()).float()
    for _ in range(config['max_trajectory']): 
        with torch.no_grad():
            env.render()
            dist = torch.distributions.normal.Normal(
                        model(ob), torch.exp(model.std))
            action = dist.sample()
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float()
            if done: 
                break
    env.close()

if __name__ == '__main__': 
    main()
