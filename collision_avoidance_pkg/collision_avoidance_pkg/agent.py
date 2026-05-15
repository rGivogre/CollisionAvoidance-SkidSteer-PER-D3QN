import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters from the paper and standard RL practices
STATE_SIZE = 50
ACTION_SIZE = 11
HIDDEN_NEURONS = 300
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.00025
MIN_EPSILON = 0.05
DECAY_RATE_BETA = 0.999  
TARGET_UPDATE_FREQ = 1000 # Steps between target network updates

class QNetwork(nn.Module):
    """
    Neural Network Architecture as described in the paper:
    Two hidden layers, each with 300 neurons and ReLU activation.
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.fc3 = nn.Linear(HIDDEN_NEURONS, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Policy Network and Target Network
        self.policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.device)
        self.target_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.device)
        
        # Copy weights from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Experience Replay Memory
        self.memory = deque(maxlen=100000)
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.step_count = 0

    def get_action(self, state):
        """
        Selects an action using the epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return random.randrange(ACTION_SIZE)
            
        # Exploitation: choose action with max Q-value
        # Note: We can optionally normalize the state here if needed in the future
        # e.g., state_tensor = torch.FloatTensor(state / 5.0).unsqueeze(0).to(self.device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Samples a mini-batch from memory and performs a gradient descent step 
        using the Double DQN logic.
        """
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Sample random mini-batch
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.device)

        # 1. Get current Q-values from Policy Net
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # 2. DDQN Logic: Select best action from Policy Net, evaluate it using Target Net
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions)
        
        # 3. Calculate target Q-values
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # 4. Calculate Loss (Mean Squared Error)
        loss = nn.MSELoss()(curr_Q, expected_Q.detach())

        # 5. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        
        # 6. Replace target network parameters every N steps
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Decays epsilon by beta. To be called at the end of each epoch.
        """
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA