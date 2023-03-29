import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from discreteaction_pendulum import Pendulum
from dqn import DQNAgent

def main():
    # Create the environment
    env = Pendulum()

    # Set the hyperparameters
    batch_size = 128
    gamma = 0.95
    target_update = 10
    num_episodes = 100

    # Create the DQN agent
    agent = DQNAgent(env.num_states, 64, env.num_actions, batch_size=batch_size, gamma=gamma)

    # Initialize the rewards and losses
    rewards = []
    losses = deque(maxlen=100)

    # Loop over episodes
    for i_episode in range(num_episodes):
        # Reset the environment and get the initial state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        # Initialize the episode reward
        episode_reward = 0

        # Loop over time steps
        for t in range(env.max_num_steps):
            # Select an action
            action = agent.select_action(state)
            
            # Take a step in the environment
            next_state, reward, done = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float)
            
            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state
            
            # Update the episode reward
            episode_reward += reward.item()
            
            # Perform one step of the optimization
            agent.optimize_model()

            if done:
                break

        # Update the target network
        if i_episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Store the episode reward
        rewards.append(episode_reward)

        #print(f'Episode {i_episode}: reward={episode_reward:.2f}')

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Curve')
        plt.savefig('figures/plot_learning_curve.png')

if __name__ == '__main__':
    main()