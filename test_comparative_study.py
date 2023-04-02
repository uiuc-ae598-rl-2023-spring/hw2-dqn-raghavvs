import random
import time
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from discreteaction_pendulum import Pendulum
from dqn import DQNAgent

start_time = time.time()

# Define constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Create the environment
    env = Pendulum()

    # Set the hyperparameters
    input_size = 64
    batch_sizes = [32, 64, 128]
    gammas = [0.9, 0.95, 0.99]
    target_update = 10
    num_episodes = 100
    max_num_steps = 200

    # Create the DQN agent
    for batch_size in batch_sizes:
        for gamma in gammas:
            agent = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update)
            policy = agent.get_policy_net()
            rewards = agent.train(env, num_episodes)

            # Plot learning curve
            plt.plot(rewards, label=f'batch_size={batch_size}, gamma={gamma}')

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.ylim(bottom=0)
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('figures/batch_size_gamma_learning_curve.png')

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()