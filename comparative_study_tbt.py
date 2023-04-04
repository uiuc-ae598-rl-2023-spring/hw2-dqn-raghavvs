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
    hidden_sizes = [32, 64, 128]
    batch_sizes = [32, 64, 128]
    gammas = [0.1, 0.5, 0.9]
    learning_rates = [1e-4, 1e-3, 1e-2]
    epsilon = [0.1, 0.5, 0.9]
    target_update = 10
    num_episodes = 100
    max_num_steps = 200

    # Create the DQN agent
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for gamma in gammas:
                for learning_rate in learning_rates:
                    for epsilon_decay in epsilon:
                        agent = DQNAgent(env.num_states, hidden_size, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update, epsilon=epsilon, learning_rate=learning_rate)
                        policy = agent.get_policy_net()
                        rewards = agent.train(env, num_episodes)

                        # Plot learning curve
                        plt.plot(rewards, label=f'hidden_size={hidden_size}, batch_size={batch_size}, gamma={gamma}, lr={learning_rate}, eps_decay={epsilon_decay}')

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.ylim(bottom=0)
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('figures/hyperparameters_learning_curve.png')

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
