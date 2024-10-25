import numpy as np
import matplotlib.pyplot as plt
import os

rewards_e30k = np.load('plots/LunarLander-v3_rewards_LunarLander-v3_epsilon_decay_30000.npy')
rewards_e50k = np.load('plots/LunarLander-v3_rewards_LunarLander-v3_hidden_size_256.npy')
rewards_e70k = np.load('plots/LunarLander-v3_rewards_LunarLander-v3_epsilon_decay_70000.npy')

def calculate_r100(rewards):
   r100_values = []
   for i in range(len(rewards)):
       if i < 99:
           r100_values.append(np.mean(rewards[:i+1]))
       else:
           r100_values.append(np.mean(rewards[i-99:i+1]))
   return r100_values

plt.figure(figsize=(12, 6))

for rewards, label in [(rewards_e30k, 'decay=30k'),
                     (rewards_e50k, 'decay=50k'),
                     (rewards_e70k, 'decay=70k')]:
   plt.plot(rewards, alpha=0.3)
   plt.plot(calculate_r100(rewards), label=label)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Epsilon Decay Comparison - LunarLander-v3')
plt.legend()
plt.grid(True, alpha=0.3)

plt.axhline(y=200, color='r', linestyle='--', label='Target Reward (200)')

plt.savefig('plots/epsilon_decay_comparison_LunarLander.pdf')
plt.close()