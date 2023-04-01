import random
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from discreteaction_pendulum import Pendulum
from dqn import DQNAgent

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
    agent = DQNAgent(env.num_states, 64, env.num_actions, batch_size=batch_size, gamma=gamma, max_num_steps=max_num_steps, target_update=target_update)
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
    plt.savefig('figures/plot_learning_curve.png') 

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
        fig.savefig('figures/Trajectory.png')

    plot_traj(env, policy)

    ###---PLOT-3: Animated gif of an example trajectory---###

    #env.video(policy, filename='figures/train_discreteaction_pendulum.gif')

    ###---PLOT-4: Policy---###

    theta_range = torch.linspace(-env.max_theta_for_upright, env.max_theta_for_upright, 100)
    thetadot_range = torch.linspace(-env.max_thetadot_for_init, env.max_thetadot_for_init, 100)
    A = torch.zeros((len(theta_range), len(thetadot_range)))
    s = None
    for i in range(len(theta_range)):
        for j in range(len(thetadot_range)):
            s = torch.from_numpy(np.array([theta_range[i], thetadot_range[j]])).float().unsqueeze(0).to(device)
            A[i, j] = policy(s)
    plt.pcolormesh(theta_range.numpy(), thetadot_range.numpy(), A.numpy())
    plt.xlabel('Theta')
    plt.ylabel('ThetaDot')
    plt.title('Policy')
    plt.colorbar()
    plt.savefig('figures/policy.png')
 
    """###---PLOT-5: State-value function---###

    def plot_policy(env, policy):
    theta = np.linspace(-np.pi, np.pi, 100)
    thetadot = np.linspace(-15, 15, 100)
    action = np.zeros([len(theta), len(thetadot)])
    
    for i in range(len(theta)):  
        for j in range(len(thetadot)): 
            s = torch.tensor([theta[i], thetadot[j]]).float()
            a = torch.argmax(policy(s)).detach()
            action[i,j] = env._a_to_u(a)
    
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(theta, thetadot, action, alpha = .75)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\dot{\theta}$')
    cbar = fig2.colorbar(c)
    cbar.ax.set_ylabel(r'$\tau$')
    fig2.savefig('./figures/policy.png')

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
    plt.savefig('figures/plot_state_value_function.png') """

###---Ablation Study---###

# Condition 1: with replay, with target Q
# Condition 2: with replay, without target Q
# Condition 3: without replay, with target Q
# Condition 3: without replay, without target Q

# Need to create a table


if __name__ == '__main__':
    main()




""" plt.figure()
plt.plot(range(num_episodes), rewards)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Learning Curve')
plt.savefig('figures/plot_learning_curve.png')
#plt.show()

state = env.reset()
done = False
states = [state]
while not done:
    action = agent.select_action(torch.tensor(state, dtype=torch.float).unsqueeze(0))
    state, reward, done = env.step(action.item())
    states.append(state)

states = np.array(states)
plt.figure()
plt.plot(states[:, 0], states[:, 1])
plt.xlabel('Theta')
plt.ylabel('ThetaDot')
plt.title('Example Trajectory')
plt.savefig('figures/plot_trajectory.png') 
#plt.show()   """   

""" Rewards:
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