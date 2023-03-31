import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.pop(0)
            self.memory.append(transition)

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, num_actions, eps):
    if torch.rand(1)[0] < eps:
        return torch.randint(num_actions, (1, 1)).type(torch.LongTensor)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)


def optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    state_action_values = policy_net(states).gather(1, actions)
    next_state_values = target_net(next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) * (1 - dones) + rewards

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
    def compute_loss(batch_size):
        state, action, reward, next_state, done = ReplayBuffer.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_values = model(state)
        next_q_values = model(next_state)

        q_value = q_values.gather(1, action)
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def train(env, num_episodes, max_steps_per_episode, batch_size, gamma):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = select_action(state)
                next_state, reward, done, _ = env.step(action.item())
                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if len(replay_buffer) > batch_size:
                    loss = compute_loss(batch_size)

            if episode % 10 == 0:
                print("Episode: {}, total reward: {}".format(episode, total_reward))

            if episode % 50 == 0:
                torch.save(model.state_dict(), "model.pth")

        env.close()







def main(input_size, hidden_size, output_size, batch_size, gamma, env, num_episodes, max_steps_per_episode):
    policy_net = DQN(input_size, hidden_size, output_size)
    target_net = DQN(input_size, hidden_size, output_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        eps = max(0.01, 1 - episode / 500)
        for t in range(max_steps_per_episode):
            action = select_action(state, policy_net, output_size, eps)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            optimize_model(memory, policy_net, target_net, optimizer, gamma, batch
