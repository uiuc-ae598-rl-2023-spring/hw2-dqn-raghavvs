import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from discreteaction_pendulum import Pendulum
from dqn import DQNAgent

def main():
    # Set up the DQN agent
    env = Pendulum()
    input_size = env.num_states
    output_size = env.num_actions
    hidden_size = 64
    batch_size = 128
    gamma = 0.95
    num_episodes = 100
    max_steps_per_episode = 1000
    agent = DQNAgent(input_size, hidden_size, output_size, batch_size, gamma, env, num_episodes, max_steps_per_episode)
    
    # Train the DQN agent
    total_rewards = agent.train_dqn()
    
    # Plot the results
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == '__main__':
    main()


""" def main(input_size, hidden_size, output_size, batch_size, gamma, env, num_episodes, max_steps_per_episode):
    agent = DQNAgent(input_size, hidden_size, output_size, batch_size, gamma, env, num_episodes, max_steps_per_episode)

    episode_lengths = []

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        episode_length = 0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state)

            if next_state is not None:
                next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)

            reward = torch.tensor([reward], dtype=torch.float32)

            if done:
                agent.memory.push(state, action, None, reward)
                episode_lengths.append(episode_length)
                break

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            episode_length += 1
            if episode_length == max_steps_per_episode:
                episode_lengths.append(episode_length)
                break

            agent.optimize_model()

        agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {i_episode + 1} finished after {episode_length} timesteps")

    plot_learning_curve(episode_lengths)

def plot_learning_curve(episode_lengths):
    episode_groups = [episode_lengths[i:i+10] for i in range(0, len(episode_lengths), 10)]
    group_lengths = [len(group) for group in episode_groups]
    group_means = [np.mean(group) for group in episode_groups]

    plt.plot(np.cumsum(group_lengths), group_means)
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Length')
    plt.show()

env = Pendulum()
main(input_size=env.num_states, hidden_size=16, output_size=env.num_actions, batch_size=128, gamma=0.95, env=env, num_episodes=100, max_steps_per_episode=200) """