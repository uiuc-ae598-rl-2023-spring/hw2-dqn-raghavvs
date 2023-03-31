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
    input_size = 64
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

        ########################################
        ############----PLOTS----###############
        ########################################

        ###---PLOT-1: Learning curve - Return vs. Number of Episodes---###

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Learning Curve')
        plt.savefig('figures/plot_learning_curve.png')

        ###---PLOT-2: Example trajectory---###

        # Define a policy that maps every state to the "zero torque" action
        policy = lambda s: env.num_actions // 2
        s = env.reset()
        s_traj = [s]
        done = False
        while not done:
            (s, r, done) = env.step(policy(s))
            s_traj.append(s)
        s_traj = torch.tensor(s_traj)
        plt.plot(s_traj[:, 0].numpy(), s_traj[:, 1].numpy())
        plt.xlabel('Theta')
        plt.ylabel('ThetaDot')
        plt.title('Example Trajectory')
        plt.savefig('figures/plot_trajectory.png')  

        ###---PLOT-3: Animated gif of an example trajectory---###

        # Simulate an episode and save the result as an animated gif
        env.video(policy, filename='figures/train_discreteaction_pendulum.gif')

        ###---PLOT-4: Policy---###

        theta_range = torch.linspace(-env.max_theta_for_upright, env.max_theta_for_upright, 100)
        thetadot_range = torch.linspace(-env.max_thetadot_for_init, env.max_thetadot_for_init, 100)
        THETA, THETADOT = torch.meshgrid(theta_range, thetadot_range)
        A = torch.zeros_like(THETA)
        for i in range(THETA.shape[0]):
            for j in range(THETA.shape[1]):
                s = torch.tensor([THETA[i, j], THETADOT[i, j]])
                A[i, j] = policy(s)
        plt.pcolormesh(THETA.numpy(), THETADOT.numpy(), A.numpy())
        plt.xlabel('Theta')
        plt.ylabel('ThetaDot')
        plt.title('Policy')
        plt.colorbar()
        plt.savefig('figures/plot_policy.png') 

        ###---PLOT-5: State-value function---###

        theta_range = torch.linspace(-env.max_theta_for_upright, env.max_theta_for_upright, 100)
        thetadot_range = torch.linspace(-env.max_thetadot_for_init, env.max_thetadot_for_init, 100)
        THETA, THETADOT = torch.meshgrid(theta_range, thetadot_range)
        V = torch.zeros_like(THETA)
        for i in range(THETA.shape[0]):
            for j in range(THETA.shape[1]):
                s = torch.tensor([THETA[i, j], THETADOT[i, j]]).unsqueeze(0)
                with torch.no_grad():
                    V[i, j] = agent.policy_net(s).max().item()
        plt.pcolormesh(THETA.numpy(), THETADOT.numpy(), V.numpy())
        plt.xlabel('Theta')
        plt.ylabel('ThetaDot')
        plt.title('State-Value Function')
        plt.colorbar()
        plt.savefig('figures/plot_state_value_function.png') 

        ###---Ablation Study---###

        # Condition 1: with replay, with target Q
        # Condition 2: with replay, without target Q
        # Condition 3: without replay, with target Q
        # Condition 3: without replay, without target Q

        # Need to create a table


if __name__ == '__main__':
    main()