import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math

# Initialize device properly
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
print(f"Using device: {device}")

num_episodes = 2000
d_state = 6
action_size = 4
discount_rate = 0.99
learning_rate = 3e-3
eps_start = 1.0
eps_end = 0.01
eps_decay = 500

# Epsilon decay function
def epsilon_by_episode(episode):
    return eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.to(device)  # Move the model to the specified device
        
        # Initialize weights using Kaiming He initialization for ReLU
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity='relu')

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

class GridGame:
    def __init__(self, model):
        self.state_size = d_state ** 2
        self.action_size = action_size
        self.model = model
        self.reset()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def reset(self):
        self.player_pos = (random.randint(0, d_state - 1), random.randint(0, d_state - 1))
        self.goal_pos = (random.randint(0, d_state - 1), random.randint(0, d_state - 1))
        while self.goal_pos == self.player_pos:
            self.goal_pos = (random.randint(0, d_state - 1), random.randint(0, d_state - 1))
        self.done = False
        self.state = self.get_state()

    def get_state(self):
        state = torch.zeros((d_state, d_state), device=device)
        state[self.player_pos[0], self.player_pos[1]] = 1
        state[self.goal_pos[0], self.goal_pos[1]] = 2
        return state.flatten().unsqueeze(0)

    def calculate_distance(self):
        # Convert the differences to tensors before calculating the distance
        a = torch.tensor((self.player_pos[0] - self.goal_pos[0])**2, device=device, dtype=torch.float)
        b = torch.tensor((self.player_pos[1] - self.goal_pos[1])**2, device=device, dtype=torch.float)
        return torch.sqrt(a + b)

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        move = moves[action]
        self.player_pos = ((self.player_pos[0] + move[0]) % d_state, (self.player_pos[1] + move[1]) % d_state)
        new_distance = self.calculate_distance()
        reward = -1  # Penalize each step to encourage efficiency
        if self.player_pos == self.goal_pos:
            reward += 100  # Large reward for reaching the goal
            self.done = True
        new_state = self.get_state()
        return new_state, reward, self.done

    def train_step(self, state, action, reward, next_state, done):
        action = action.view(1, -1)
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        done = torch.tensor([done], device=device, dtype=torch.float)
        
        state_action_values = self.model(state).gather(1, action)
        next_state_values = self.model(next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * discount_rate) * (1 - done) + reward
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def select_action(state, policy_net, episode):
    eps_threshold = epsilon_by_episode(episode)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)

policy_net = DQN(d_state ** 2, action_size)
game = GridGame(policy_net)

start_time = time.time()

try:
    for episode in range(num_episodes):
        game.reset()
        total_reward = 0
        while not game.done:
            state = game.state
            action = select_action(state, policy_net, episode)
            next_state, reward, done = game.step(action.item())
            game.train_step(state, action, reward, next_state, done)
            game.state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}")

except KeyboardInterrupt:
    print("Training stopped")

finally:
    print(f'Training took {time.time() - start_time} seconds')
    torch.save(policy_net.state_dict(), 'weights/model-v0.pth')
    print('Model saved')
