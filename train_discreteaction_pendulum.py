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
    batch_size = 128
    gamma = 0.95
    target_update = 10
    num_episodes = 100
    max_num_steps = 200

    # Create the DQN agent
    agent = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update)
    policy = agent.get_policy_net()
    rewards = agent.train(env, num_episodes)
    
    ########################################
    ############----PLOTS----###############
    ########################################

    ###---PLOT-1: Learning curve - Return vs. Number of Episodes---###

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.ylim(bottom=0)
    plt.title('Learning Curve')
    plt.savefig('figures/learning_curve.png') 

    ###---PLOT-2: Example trajectory---###

    def plot_traj(env, policy): 
        # Initialize simulation
        s = env.reset()
        policy = agent.get_policy_net()

        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        # Simulate until episode is done
        done = False
        rew =  0 
        while not done:
            a = torch.argmax(policy(torch.tensor(s).float()))
            (s, r, done) = env.step(a)
            rew += r
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(rew)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(data['t'], theta, label='theta')
        ax[0].plot(data['t'], thetadot, label='thetadot')
        ax[0].legend()
        ax[1].plot(data['t'][:-1], tau, label='tau')
        ax[1].legend()
        ax[2].plot(data['t'][:-1], data['r'], label='return')
        ax[2].legend()
        ax[2].set_xlabel('time step')
        plt.tight_layout()
        fig.savefig('figures/trajectory.png')

    plot_traj(env, policy)

    ###---PLOT-3: Animated gif of an example trajectory---###

    #env.video(policy, filename='figures/train_discreteaction_pendulum.gif')

    ###---PLOT-4: Policy---###

    theta_range = torch.linspace(-env.max_theta_for_upright, env.max_theta_for_upright, 100)
    thetadot_range = torch.linspace(-env.max_thetadot_for_init, env.max_thetadot_for_init, 100)
    A = torch.zeros((len(theta_range), len(thetadot_range)))
    for i in range(len(theta_range)): 
        for j in range(len(thetadot_range)): 
            s = torch.tensor([theta_range[i], thetadot_range[j]]).float()
            a = torch.argmax(policy(s)).detach()
            A[i,j] = env._a_to_u(a)
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(theta_range, thetadot_range, A, alpha = .75)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\dot{\theta}$')
    cbar = fig2.colorbar(c)
    cbar.ax.set_ylabel(r'$\tau$')
    plt.title('Policy')
    fig2.savefig('figures/policy.png')
    

    ###---PLOT-5: State-value function---###

    theta_range = torch.linspace(-env.max_theta_for_upright, env.max_theta_for_upright, 100)
    thetadot_range = torch.linspace(-env.max_thetadot_for_init, env.max_thetadot_for_init, 100)
    value = torch.zeros([len(theta_range), len(thetadot_range)])
    for i in range(len(theta_range)): 
        for j in range(len(thetadot_range)): 
                s = torch.tensor([theta_range[i], thetadot_range[j]]).float()
                v = torch.max(policy(s)).detach()
                value[i,j] = v
    fig3, ax3 = plt.subplots()
    c = ax3.contourf(theta_range, thetadot_range, value, alpha = .75)
    ax3.set_xlabel(r'$\theta$')
    ax3.set_ylabel(r'$\dot{\theta}$')
    cbar = fig3.colorbar(c)
    cbar.ax.set_ylabel('value')
    fig3.savefig('figures/state_value_function.png')
    
    ###---Ablation Study---###

    # Condition 1: with replay, with target Q
    agent1 = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update)
    rewards1 = agent1.train(env, num_episodes=100)

    # Condition 2: with replay, without target Q
    agent1 = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=float('inf'))
    rewards2 = agent1.train(env, num_episodes=100)

    # Condition 3: without replay, with target Q
    agent1 = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=1, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update)
    rewards3 = agent1.train(env, num_episodes=100)

    # Condition 3: without replay, without target Q
    agent1 = DQNAgent(env.num_states, input_size, env.num_actions, batch_size=1, gamma=gamma, max_num_steps=max_num_steps, target_update=float('inf'))
    rewards4 = agent1.train(env, num_episodes=100)

    plt.figure()
    plt.plot(rewards1)
    plt.plot(rewards2)
    plt.plot(rewards3)
    plt.plot(rewards4)
    plt.legend(['with replay, with target Q', 'with replay, without target Q', 'without replay, with target Q', 'without replay, without target Q'])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.ylim([0, 40])
    plt.title('Learning Curve')
    plt.savefig('figures/ablation_study_learning_curve.png') 


if __name__ == '__main__':
    main()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


""" 
Time taken for 100 episodes:  289 seconds
Time taken for 300 episodes:  289 seconds
Time taken: 4969.680328845978 seconds

Rewards:
Episode 89: reward=14.00
Episode 90: reward=15.00
Episode 91: reward=19.00
Episode 92: reward=88.00
Episode 93: reward=11.00
Episode 94: reward=15.00
Episode 95: reward=84.00
Episode 96: reward=11.00
Episode 97: reward=11.00
Episode 98: reward=10.00
Episode 99: reward=81.00

Episode 989: reward=87.00
Episode 990: reward=91.00
Episode 991: reward=90.00
Episode 992: reward=97.00
Episode 993: reward=100.00
Episode 994: reward=78.00
Episode 995: reward=87.00
Episode 996: reward=88.00
Episode 997: reward=88.00
Episode 998: reward=94.00
Episode 999: reward=87.00 """