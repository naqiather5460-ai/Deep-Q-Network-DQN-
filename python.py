import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import random

# --- 1. Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Slimmer Neural Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # <-- CHANGE 1
        self.fc2 = nn.Linear(128, 128)        # <-- CHANGE 1
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- 3. Experience Replay Buffer (Unchanged) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- 4. DQN Agent with Tweaked Hyperparameters ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(capacity=100000)
        
        # Hyperparameters (with changes)
        self.gamma = 0.99
        self.batch_size = 32              # <-- CHANGE 3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.target_update_freq = 100     # <-- CHANGE 3
        self.train_step_counter = 0

    def select_action(self, state, evaluate=False):
        if evaluate:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# --- 5. Training Loop with Linear Epsilon Decay ---
def train_dqn(env, agent, episodes=500, target_score=475):
    scores = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        # <-- CHANGE 2: Linear Epsilon Decay over 400 episodes -->
        agent.epsilon = max(agent.epsilon_min, 1.0 - (episode / 400))

        scores.append(total_reward)
        
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {episode:3d} | Avg Reward (last 20): {avg_score:.1f} | Epsilon: {agent.epsilon:.3f}")

        if episode > 100 and np.mean(scores[-100:]) >= target_score:
            print(f"Solved in {episode} episodes! Average score: {np.mean(scores[-100:]):.1f}")
            break
    
    return scores

# --- 6. Main Execution ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    
    print("--- Starting Training with Tweaked Settings ---")
    scores = train_dqn(env, agent)
    env.close()

    # --- 7. Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(scores, alpha=0.6, label="Reward per episode")
    window = 20
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(scores)), moving_avg, color='red', label=f"Moving Avg ({window} eps)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance (Tweaked: Slimmer Net, Linear Eps, Batch=32)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 8. Testing ---
    print("\n--- Testing Tweaked Agent ---")
    test_env = gym.make("CartPole-v1", render_mode="human")
    test_rewards = []
    for test_ep in range(100):
        state, _ = test_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
        test_rewards.append(ep_reward)
    test_env.close()

    mean_test = np.mean(test_rewards)
    std_test = np.std(test_rewards)
    print(f"Test over 100 episodes: Mean = {mean_test:.2f}, Std = {std_test:.2f}")
    print(f"Max: {max(test_rewards)}, Min: {min(test_rewards)}")
    print("Environment solved if mean >= 475.0")