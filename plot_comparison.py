import numpy as np
import matplotlib.pyplot as plt
import os

rewards_lr_0001 = np.load('plots/LunarLander-v3_rewards_LunarLander-v3_learning_rate_1e-4.npy')
rewards_lr_001 = np.load('plots/LunarLander-v3_rewards_LunarLander-v3_learning_rate_1e-3.npy')

def calculate_r100(rewards):
    r100_values = []
    for i in range(len(rewards)):
        if i < 99:
            r100_values.append(np.mean(rewards[:i+1]))
        else:
            r100_values.append(np.mean(rewards[i-99:i+1]))
    return r100_values

plt.figure(figsize=(12, 6))

for rewards, label in [(rewards_lr_0001, 'lr=1e-4'),
                      (rewards_lr_001, 'lr=1e-3'),
                      ]:
    plt.plot(rewards, alpha=0.3)
    plt.plot(calculate_r100(rewards), label=label)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Rate Comparison - LunarLander-v3')
plt.legend()
plt.grid(True, alpha=0.3)

plt.axhline(y=200, color='r', linestyle='--', label='Target Reward (200)')

plt.savefig('plots/learning_rate_comparison_LunarLander.pdf')
plt.close()