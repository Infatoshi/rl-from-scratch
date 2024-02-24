import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math
import pygame


# Initialize device properly
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
print(f"Using device: {device}")

num_episodes = 10000
d_state = 5
action_size = 4
discount_rate = 0.8
learning_rate = 5e-4
eps_start = 1
eps_end = 0.01
eps_decay = 2000
time_step_reward = -1
dropout = 0.3
r_scaling = 2

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

        self.dropout = nn.Dropout(p=dropout)
        # Initialize weights using Kaiming He initialization for ReLU
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity='relu')

    def forward(self, x):
        out = self.dropout(self.net(x))
        return out

    
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
        prev_distance = self.calculate_distance()
        self.player_pos = ((self.player_pos[0] + move[0]) % d_state, (self.player_pos[1] + move[1]) % d_state)
        new_distance = self.calculate_distance()

        reward = time_step_reward  # Penalize each step to encourage efficiency
        delta_distance = prev_distance - new_distance

        if delta_distance > 0:
            reward += delta_distance/d_state
        else:
            reward -= delta_distance/d_state

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

        expected_state_action_values = (next_state_values * discount_rate) * (1 - done) + torch.tanh(reward.clone().detach().requires_grad_(False))*r_scaling
        
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
policy_net.train()

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
            print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {epsilon_by_episode(episode):.2f}")

except KeyboardInterrupt:
    print("Training stopped")

finally:
    print(f'Training took {time.time() - start_time} seconds')
    torch.save(policy_net.state_dict(), 'weights/model-v0.pth')
    print('Model saved')


policy_net.eval()
episode_moves = []

for x in range(100):
    game.reset()
    screen = pygame.display.set_mode((d_state*100, d_state*100))
    total_reward = 0
    episode_moves.append(0)
    while not game.done:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        state = game.state
        human_readable_state = state.view(d_state, d_state).cpu().numpy()
        action = select_action(state, policy_net, 10000)
        next_state, reward, done = game.step(action.item())
        game.state = next_state
        total_reward += reward
        episode_moves[x] += 1

        time.sleep(0)
        screen.fill((0, 0, 0))
        for i in range(d_state):
            for j in range(d_state):
                if human_readable_state[i][j] == 1:
                    pygame.draw.rect(screen, (0, 0, 255), (i*100, j*100, 100, 100))
                elif human_readable_state[i][j] == 2:
                    pygame.draw.rect(screen, (0, 255, 0), (i*100, j*100, 100, 100))
        pygame.display.update()
    print(f"Total Reward: {total_reward}")
pygame.quit()
print(f"Average moves: {sum(episode_moves)/len(episode_moves)}")