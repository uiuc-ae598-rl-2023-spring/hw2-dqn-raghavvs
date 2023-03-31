####-----DESCRIPTION------####

""" 
DQN algorithm with experience replay:

Initialize the neural network with random weights theta and the target network with the same weights theta’.
Initialize the replay memory buffer with a fixed capacity.
For each episode:
    Initialize the state s.
    For each step:
        Select an action a using an epsilon-greedy exploration strategy. 
        Execute the action a and observe the next state s’ and reward r.
        Store the transition (s, a, s’, r) in the replay memory buffer.
        Sample a random batch of transitions from the replay memory buffer.
        For each transition in the batch:
            Compute the target value y using the target network:

            y = r + gamma * max Q(s’, a’; theta’)

            If s’ is terminal, then y = r.

            Compute the prediction value Q(s, a; theta) using the neural network.

            Compute the loss between y and Q(s, a; theta) using a suitable loss function such as mean squared error or Huber loss.

        Perform one step of gradient descent to update theta using backpropagation.
        Every N steps, copy theta to theta’ to synchronize the target network with the neural network.
        Update s to s’.
Repeat until s is terminal or maximum number of steps is reached. 
"""

####-----IMPORT LIBRARIES------####

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# Define the experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, batch_size, gamma, env, num_episodes, max_steps_per_episode):
        self.policy_net = DQN(input_size, hidden_size, output_size)
        self.target_net = DQN(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        
        self.memory = ReplayMemory(10000)
        
        self.batch_size = batch_size
        self.gamma = gamma

    def select_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

    def train_dqn(self):
        total_rewards = []
        episode_rewards = []
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                # Select and perform an action
                action = self.agent.select_action(torch.FloatTensor([state]))
                next_state, reward, done, _ = env.step(action.item())
                
                # Store the transition in the replay memory
                self.agent.memory.push(torch.FloatTensor([state]),
                                action,
                                torch.FloatTensor([next_state]) if not done else None,
                                torch.FloatTensor([reward]))
                
                # Update the DQN agent
                self.agent.optimize_model()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward = sum(episode_rewards)
            total_rewards.append(total_reward)
            print("Episode {}: reward={}".format(episode, episode_reward))
            
            # Update the target network every 10 episodes
            if episode % 10 == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                
        return total_rewards
