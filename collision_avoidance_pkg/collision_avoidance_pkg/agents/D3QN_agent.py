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
LEARNING_RATE = 2.5e-4
MIN_EPSILON = 0.05
DECAY_RATE_BETA = 0.999  
TARGET_UPDATE_FREQ = 1000 # Steps between target network updates

class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture:
    Shares two hidden layers, then splits into separate Value (V) 
    and Advantage (A) streams before aggregating them into Q-values.
    """
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        
        # Shared Feature Extraction Layers
        self.fc1 = nn.Linear(state_size, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        
        # CHANGED: Value Stream - Outputs a single scalar value for the state V(s)
        self.value_stream = nn.Linear(HIDDEN_NEURONS, 1)
        
        # CHANGED: Advantage Stream - Outputs an advantage value for each action A(s, a)
        self.advantage_stream = nn.Linear(HIDDEN_NEURONS, action_size)

    def forward(self, x):
        # Extract features through shared network layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Calculate Value and Advantage separately
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # CHANGED: Combine streams using the standard mean-subtraction formula:
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # This enforces identifiability so both streams learn their unique definitions.
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values


class DuelingDDQNAgent:
    def __init__(self):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CHANGED: Initialize Policy and Target Networks with the Dueling Architecture
        self.policy_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   
        self.target_net.eval()  
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    
        self.memory = deque(maxlen=100000)  # Standard Experience Replay Memory
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.step_count = 0

    def get_action(self, state):
        """ Selects an action using the epsilon-greedy policy. """
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            # Dueling network outputs Q-values identically, meaning action selection logic is unchanged
            q_values = self.policy_net(state_tensor)    
        
        return torch.argmax(q_values).item()    

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Samples a mini-batch from memory and performs a gradient descent step 
        using Double DQN evaluation on top of the Dueling architecture.
        """
        if len(self.memory) < BATCH_SIZE:   
            return
            
        # Sample random mini-batch
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.train_device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.train_device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.train_device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.train_device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.train_device)

        # Get current Q-values from Dueling Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # DDQN logic: Select best action from Policy Network, evaluate it using Target Network
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions) 
        
        # Calculate target Q-values 
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # Calculate Loss using Huber (Smooth L1) for stable learning
        loss = nn.SmoothL1Loss()(expected_Q.detach(), curr_Q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        
        # Replace target network parameters every N steps
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA
            if self.epsilon < MIN_EPSILON:
                self.epsilon = MIN_EPSILON