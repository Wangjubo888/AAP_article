import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm1d(400)
        self.batch_norm2 = nn.BatchNorm1d(300)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if input is 1D
        x = self.fc1(state)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return self.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400 + action_size, 300)
        self.fc3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(400)

    def forward(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if input is 1D
        xs = self.fc1(state)
        xs = self.batch_norm1(xs)
        xs = self.relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)