import argparse
import os

import gymnasium as gym
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from dqn_common import epsilon_by_frame, DqnNetSingleLayer, DqnNetTwoLayers, alpha_sync, DuellingDqn
from lib.experience_buffer import ExperienceBuffer, Experience
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="CartPole-v1",
                    help="Full name of the environment, e.g. CartPole-v1, LunarLander-v3, etc.")
parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")
parser.add_argument("-n", "--network", default='s',
                    help="DQN network architecture `single-hidden` for single hidden layer, `two-hidden` for 2 hidden layers and `duelling-dqn` for duelling DQN",
                    choices=['single-hidden', 'two-hidden', 'duelling-dqn'])
parser.add_argument("-s", "--seed", type=int, help="Manual seed (leave blank for random seed)")
args = parser.parse_args()


def calculate_r100(rewards):

    r100_values = []
    for i in range(len(rewards)):
        if i < 99:
            r100_values.append(np.mean(rewards[:i + 1]))
        else:
            r100_values.append(np.mean(rewards[i - 99:i + 1]))
    return r100_values


def plot_training(env_name, rewards, save_dir='plots'):

    r100_values = calculate_r100(rewards)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(r100_values)), r100_values)
    plt.xlabel('Episode')
    plt.ylabel('R100 (100-episode moving average reward)')
    plt.title(f'Training Progress - {env_name}')
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{env_name}_training_LunarLander-v3_epsilon_decay_70000.pdf'))
    np.save(os.path.join(save_dir, f'{env_name}_rewards_LunarLander-v3_epsilon_decay_70000.npy'), np.array(rewards))
    plt.close()


# Hyperparameters for the requried environment
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

if args.env not in hypers:
    raise Exception(
        f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/dqn.yaml)')
params = hypers[args.env]

env = gym.make(args.env)

# Set seeds
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU")
else:
    device = torch.device("cpu")
    print("Training on CPU")

if args.network == 'two-hidden':
    net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                          hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                          n_actions=env.action_space.n).to(device)
    target_net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                                 hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                                 n_actions=env.action_space.n).to(device)
elif args.network == 'single-hidden':
    net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                            hidden_size=params['hidden_size'],
                            n_actions=env.action_space.n).to(device)
    target_net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                                   hidden_size=params['hidden_size'],
                                   n_actions=env.action_space.n).to(device)
else:
    net = DuellingDqn(obs_size=env.observation_space.shape[0],
                      hidden_size=params['hidden_size'],
                      n_actions=env.action_space.n).to(device)
    target_net = DuellingDqn(obs_size=env.observation_space.shape[0],
                             hidden_size=params['hidden_size'],
                             n_actions=env.action_space.n).to(device)

print(net)

buffer = ExperienceBuffer(int(params['replay_size']), device)

optimizer = optim.Adam(net.parameters(), lr=float(params['learning_rate']))

frame_idx = 0
max_reward = -math.inf
all_rewards = []
losses = []
episode_reward = 0
r100 = -math.inf
episode_start = time.time()
start = time.time()
episode_frame = 0
episode_no = 0
visualizer_on = False

state, _ = env.reset()


def calculate_loss(net, target_net):
    states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(params['batch_size'])

    Q_s = net.forward(states_v)
    state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net.forward(next_states_v).max(1)[0]
    expected_state_action_values = rewards_v + next_state_values.detach() * params['gamma'] * (1 - dones_v)
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()

    if params['clip_gradient']:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)

    optimizer.step()

    return loss


while True:
    frame_idx += 1

    # calculate the value of decaying epsilon
    epsilon = epsilon_by_frame(frame_idx, params)
    if np.random.random() < epsilon:
        # explore
        action = env.action_space.sample()
    else:
        # exploit
        state_a = np.asarray([state])
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = act_v.item()

    # take step in the environment
    new_state, reward, terminated, truncated, _ = env.step(action)
    is_done = terminated or truncated
    episode_reward += reward

    # store the transition in the experience replay buffer
    exp = Experience(state, action, reward, is_done, new_state)
    buffer.append(exp)
    state = new_state

    # when the episode is done, reset and update progress
    if is_done:
        done_reward = episode_reward
        all_rewards.append(episode_reward)
        episode_no += 1

        state, _ = env.reset()
        if episode_reward > max_reward:
            max_reward = episode_reward

        if len(all_rewards) > 10 and len(losses) > 10:
            r100 = np.mean(all_rewards[-100:])
            l100 = np.mean(losses[-100:])
            fps = (frame_idx - episode_frame) / (time.time() - episode_start)
            print(
                f"Frame: {frame_idx}: Episode: {episode_no}, R100: {r100: .2f}, MaxR: {max_reward: .2f}, R: {episode_reward: .2f}, FPS: {fps: .1f}, L100: {l100: .2f}, Epsilon: {epsilon: .4f}")

            # visualize the training when reached 95% of the target R100
            if not visualizer_on and r100 > 0.95 * params['stopping_reward']:
                env = gym.make(args.env, render_mode='human')
                env.reset()
                env.render()
                visualizer_on = True

        episode_reward = 0
        episode_frame = frame_idx
        episode_start = time.time()

    if len(buffer) < params['replay_size_start']:
        continue

    # do the learning
    loss = calculate_loss(net, target_net)
    losses.append(loss.item())

    if params['alpha_sync']:
        alpha_sync(net, target_net, alpha=1 - params['tau'])
    elif frame_idx % params['target_net_sync'] == 0:
        target_net.load_state_dict(net.state_dict())

    if r100 > params['stopping_reward']:
        print("Finished training")

        name = f"{args.env}_{args.network}_nn_DQN_act_net_%+.3f_%d.dat" % (r100, frame_idx)
        if not os.path.exists(params['save_path']):
            os.makedirs(params['save_path'])
        torch.save(net.state_dict(), os.path.join(params['save_path'], name))

        plot_training(args.env, all_rewards)
        break

    if frame_idx > params['max_frames']:
        print(f"Ran out of time at {time.time() - start}")
        plot_training(args.env, all_rewards)
        break