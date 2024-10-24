import numpy as np
import matplotlib.pyplot as plt
import os

rewards_1e4 = np.load('plots/CartPole-v0_rewards_1e-4.npy')
rewards_1e3 = np.load('plots/CartPole-v0_rewards.npy')
rewards_1e2 = np.load('plots/CartPole-v0_rewards_1e-2.npy')

def calculate_r100(rewards):
    r100_values = []
    for i in range(len(rewards)):
        if i < 99:
            r100_values.append(np.mean(rewards[:i+1]))
        else:
            r100_values.append(np.mean(rewards[i-99:i+1]))
    return r100_values

plt.figure(figsize=(10, 6))
plt.plot(calculate_r100(rewards_1e4), label='lr=0.0001', linestyle='-')
plt.plot(calculate_r100(rewards_1e3), label='lr=0.001', linestyle='-')
plt.plot(calculate_r100(rewards_1e2), label='lr=0.01', linestyle='-')

plt.xlabel('Episode')
plt.ylabel('R100 (100-episode moving average reward)')
plt.title('Learning Rate Comparison - CartPole-v0')
plt.legend()
plt.grid(True)

plt.savefig('plots/learning_rate_comparison.pdf')
plt.close()