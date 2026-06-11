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
TAU = 0.005                 # for soft update of target network

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

    def forward(self, x):               # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgent:
    def __init__(self, learning_rate=LEARNING_RATE):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        # Initialize Policy Network and Target Network
        self.policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   # Copy weights from policy network to target network
        self.target_net.eval()  # Target network is only used for inference, it is not trained during the optimization step
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
        self.memory = deque(maxlen=75000)  # Experience Replay Memory
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0

    def get_action(self, state):
        """ Selects an action using the epsilon-greedy policy. """
        if np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return random.randrange(ACTION_SIZE)
            
        # Exploitation: 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)    # get Q-values for all possible actions
        
        return torch.argmax(q_values).item()    # return index of the action with the highest Q-value

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Samples a mini-batch from memory and performs a gradient descent step 
        using the Double DQN logic.
        """
        if len(self.memory) < BATCH_SIZE:   # if we don't have enough samples in memory, skip training
            return
            
        # Sample random mini-batch
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.train_device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.train_device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.train_device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.train_device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.train_device)

        # Get current Q-values from Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)     # gather(1, actions) selects the Q-values corresponding to the actions taken
        
        # DDQN logic: Select best action from Policy Network, evaluate it using Target Network
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions)          # (theory: next_Q is the estimate of cumulative future reward from the next state s')
        
        # Calculate target Q-values i.e., the values we want our current Q-values to move towards
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))           #(dimensions: [batch_size, 1])

        # Calculate Loss (Smooth L1 Loss / Huber Loss)
        loss = nn.SmoothL1Loss()(expected_Q.detach(), curr_Q)   # expected - curr is the TD error

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
        
        self.optimizer.step()

        # Soft update of target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def update_epsilon(self):
        """
        Decays epsilon by beta. To be called at the end of each epoch.
        """
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA