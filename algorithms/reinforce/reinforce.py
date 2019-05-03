import gym
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions.normal import Normal
import os
import errno
import gym_driving
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PG')
parser.add_argument('-n', '--train-steps', default=960000, type=int, help='number of training steps')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='learning rate of the network')
parser.add_argument('-epi', '--episodes-per-iteration', default=20, type=int, help='Number of episodes per '
                                                                                       'training iteration')
parser.add_argument('-gamma', '--discount-factor', default=0.99, type=float, help='discount factor')
parser.add_argument('--no-gpu', action='store_true')
parser.add_argument('-tn', '--test_episodes', default=10, type=int, help='number of test episodes')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', default='', help='previous checkpoint from which training is to be resumed')
parser.add_argument('-env', '--environment', default='Driving-v0', help='OpenAI Mujoco environment name')
parser.add_argument('-l', '--layers', nargs='+', help='hidden layer dimensions', required=True)
parser.add_argument('-s', '--seed', default=4, type=int, help='seed')
parser.add_argument('-tlim', '--trajectory-limit', default=1200, type=int, help='maximum number of steps in a '
                                                                               'trajectory')


class Network(nn.Module):
    # Policy network takes as input the intermediate layer dimensions and outputs the means for each dimension of the
    #  action. In this code, the log of the standard deviation of the normal distribution (logstd as seen in the code
    #  below) is not part of the network output; it is a separate variable that can be trained.
    def __init__(self, input_shape, output_shape, hidden_layers):
        super(Network, self).__init__()
        modules = []
        modules.append(nn.Linear(input_shape, hidden_layers[0]))
        modules.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            modules.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_layers[-1], output_shape))
        self.sequential = nn.Sequential(*modules)
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.5)

    def forward(self, x):
        return self.sequential(x)


def find_discounted_rewards(rewards, gamma):
    batch_discounted_rewards = []
    for batch in range(len(rewards)):
        discounted_rewards = []
        running_reward = 0
        for i in range(len(rewards[batch]) - 1, -1, -1):
            running_reward = rewards[batch][i] + gamma * running_reward
            discounted_rewards.append(running_reward)
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)
        discounted_rewards = list(reversed(discounted_rewards))
        batch_discounted_rewards.append(discounted_rewards)
    return np.array(batch_discounted_rewards)


def convert_to_variable(x, gpu=True, grad=True):
    if gpu:
        return Variable(torch.cuda.FloatTensor(x), requires_grad=grad)
    return Variable(torch.FloatTensor(x), requires_grad=grad)


def test(env, net, logstd, test_episodes, no_gpu, render=False):
    # Function to check the performance of the existing network and logstd on a few test trajectories.
    average_reward = 0
    average_steps = 0
    for i in range(test_episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        steps = 0
        while not done:
            means = net(convert_to_variable(state, gpu=(not no_gpu), grad=False))
            action_dist = Normal(means, torch.exp(logstd))
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            if render:
                env.render()

        average_reward += episode_reward
        average_steps += steps
        if render:
            print('Episode reward: ', episode_reward)
    average_reward /= test_episodes
    average_steps /= test_episodes
    print('Average reward: %f' % (average_reward))
    print('Average steps: %f' % (average_steps))
    env.close()
    return average_reward


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename, net, optimizer):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    print('Loading checkpoint ', filename)
    checkpoint = torch.load(filename)
    total_steps = checkpoint['total_steps']
    total_episodes = checkpoint['total_episodes']
    net.load_state_dict(checkpoint['state_dict'])
    logstd = checkpoint['logstd']
    optimizer.add_param_group({"name": "logstd", "params": logstd})
    optimizer.load_state_dict(checkpoint['optimizer'])
    return total_steps, total_episodes, net, optimizer, logstd


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = parser.parse_args()
    num_training_steps = args.train_steps
    lr = args.learning_rate
    episodes_per_iteration = args.episodes_per_iteration
    gamma = args.discount_factor
    no_gpu = args.no_gpu
    test_episodes = args.test_episodes
    checkpoint_file = args.resume
    test_only = args.test_only
    env_name = args.environment
    layers = args.layers
    seed = args.seed
    trajectory_limit = args.trajectory_limit

    env = gym.make(env_name)
    set_global_seed(seed)
    env.seed(seed)

    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.shape[0]
    hidden_layers = [int(x) for x in layers]

    net = Network(input_shape, output_shape, hidden_layers)
    total_steps = 0
    total_episodes = 0

    no_gpu = True
    if not no_gpu:
        net = net.cuda()

    gerard_logger = []
    optimizer = Adam(net.parameters(), lr=lr)

    if checkpoint_file:
        total_steps, total_episodes, net, optimizer, logstd = load_checkpoint(checkpoint_file, net, optimizer)

    else:
        logstd = Variable(torch.ones(output_shape), requires_grad=True)
        optimizer.add_param_group({"name": "logstd", "params": logstd})

    if test_only:
        test(env, net, logstd, test_episodes, no_gpu, render=True)
        return

    # Path to the directory where the checkpoints and log file will be saved. If there is no existing directory,
    # we create one.
    checkpoint_dir = os.path.join(env_name, 'pg_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.isfile(os.path.join(checkpoint_dir, 'log.txt')):
        f = open(os.path.join(checkpoint_dir, 'log.txt'), 'a')
    else:
        f = open(os.path.join(checkpoint_dir, 'log.txt'), 'w')
        f.write('Net: ' + str(input_shape) + ', ' + ', '.join(layers) + ', ' + str(output_shape) + '\n')

    while total_steps < num_training_steps:
        batch_rewards = []
        batch_logps = []
        for j in range(episodes_per_iteration):
            state = env.reset()
            done = False
            rewards = []
            logps = []
            steps = 0

            while not done and not steps >= trajectory_limit:
                # The network outputs the mean values for each action dimension.
                means = net(convert_to_variable(state, gpu=(not no_gpu), grad=False))
                # We now have a normal distribution for each action dimension, since we have the means from the
                # network and the trainable parameter, logstd.
                action_dist = Normal(means, torch.exp(logstd))
                # We sample an action from the normal distribution.
                action = action_dist.sample()
                # We compute the log-likelihood of that action.
                logp = action_dist.log_prob(action)
                state, reward, done, info = env.step(action.cpu())
                rewards.append(reward)
                logps.append(logp)
                total_steps += 1
                steps += 1
            gerard_logger.append(sum(rewards))
            batch_rewards.append(rewards)
            batch_logps.append(logps)
            total_episodes += 1
            f.write(str(total_episodes) + ':' + str(np.array(rewards).sum()) + ':' + str(total_steps) + '\n')

        batch_discounted_rewards = find_discounted_rewards(batch_rewards, gamma)

        loss = []
        for batch in range(len(batch_discounted_rewards)):
            loss.append([])
            for (r, logp) in zip(batch_discounted_rewards[batch], batch_logps[batch]):
                # The loss for policy gradient at each time step equals the product of the negative log-likelihood of
                #  the action taken and the discounted reward at that step.
                loss[batch].append(-logp * r)

        optimizer.zero_grad()
        batch_loss = 0
        for l in loss:
            batch_loss += torch.cat(l).sum()
        batch_loss = batch_loss.mean()
        batch_loss.backward()
        optimizer.step()

        print(len(gerard_logger), ": ", round(sum(gerard_logger[-20:]) / 20, 2))

        if (total_episodes) % (episodes_per_iteration * 100) == 0 or total_steps >= num_training_steps:
            save_checkpoint(
                {'total_steps': total_steps, 'total_episodes': total_episodes, 'state_dict':net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'logstd': logstd}, filename=os.path.join(checkpoint_dir,
                                                                                               str(total_episodes)
                                                                                               +'.pth.tar'))
            print('Logstd: ', logstd)
            print('\n')

    with open('reinforce_rew_per_episode', 'wb') as fp: 
        dic = {'reinforce': gerard_logger}
        pickle.dump(dic, fp)
    avg_rew = []
    avg = 10
    for bottom_ind in range(0, len(gerard_logger), avg):
        avg_rew.append(sum(gerard_logger[bottom_ind: bottom_ind + avg]) / avg)
    graph_reward(avg_rew)

    f.close()

    return

def graph_reward(ep_reward):
    '''
    Graph info of ep_reward
    '''
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(ep_reward)
    ran = (max(ep_reward) - min(ep_reward)) * 0.1
    ax.set_ylim((min(ep_reward) - ran, max(ep_reward) + ran))
    ax.set_title('Reward - Averaged per 10 Episodes')
    plt.show()



if __name__ == '__main__':
    main()
