import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from cpprb import PrioritizedReplayBuffer  # Imported from external library

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

# PER Specific Hyperparameters
PER_ALPHA = 0.6       # Determines how much prioritization is used
PER_BETA_START = 0.4  # Importance sampling correction factor starting value
PER_BETA_STEPS = 100000 # Number of steps to anneal beta to 1.0
PER_EPSILON = 1e-5    # Small positive constant to ensure non-zero priority


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
        
        # Value Stream - Outputs a single scalar value for the state V(s)
        self.value_stream = nn.Linear(HIDDEN_NEURONS, 1)
        
        # Advantage Stream - Outputs an advantage value for each action A(s, a)
        self.advantage_stream = nn.Linear(HIDDEN_NEURONS, action_size)

    def forward(self, x):
        # Extract features through shared network layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Calculate Value and Advantage separately
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine streams using the standard mean-subtraction formula:
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values


class DuelingPERDDQNAgent:
    def __init__(self):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Dueling Policy Network and Target Network
        self.policy_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   
        self.target_net.eval()  
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    
        # Setup cpprb's Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(
            size=100000,
            env_dict={
                "state": {"shape": STATE_SIZE},
                "action": {"shape": 1, "dtype": np.int64},
                "reward": {"shape": 1},
                "next_state": {"shape": STATE_SIZE},
                "done": {"shape": 1}
            },
            alpha=PER_ALPHA
        )
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.step_count = 0
        
        # PER Beta Tracking
        self.beta = PER_BETA_START

    def get_action(self, state):
        """ Selects an action using the epsilon-greedy policy. """
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)    
        
        return torch.argmax(q_values).item()    

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the prioritized replay memory."""
        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def train_step(self):
        """ Samples a prioritized mini-batch and optimizes the Dueling network. """
        if self.memory.get_stored_size() < BATCH_SIZE:   
            return
            
        # Anneal beta towards 1.0
        self.beta = min(1.0, PER_BETA_START + self.step_count * (1.0 - PER_BETA_START) / PER_BETA_STEPS)

        # Sample mini-batch based on priority scores
        samples = self.memory.sample(BATCH_SIZE, self.beta)
        
        states = torch.FloatTensor(samples["state"]).to(self.train_device)
        actions = torch.LongTensor(samples["action"]).to(self.train_device)
        rewards = torch.FloatTensor(samples["reward"]).to(self.train_device)
        next_states = torch.FloatTensor(samples["next_state"]).to(self.train_device)
        dones = torch.FloatTensor(samples["done"]).to(self.train_device)
        
        is_weights = torch.FloatTensor(samples["weights"]).unsqueeze(1).to(self.train_device)
        idxs = samples["indexes"]

        # Get current Q-values from the Dueling Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # DDQN evaluation logic: select actions with Policy Net, evaluate with Target Net
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions) 
        
        # Calculate target Q-values 
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # Calculate individual absolute errors (TD-errors) to update priorities
        with torch.no_grad():
            td_errors = (expected_Q - curr_Q).cpu().numpy().flatten()
            
        # Update priorities in cpprb's internal tree structure
        self.memory.update_priorities(idxs, np.abs(td_errors) + PER_EPSILON)

        # Calculate loss penalizing based on Importance Sampling weights
        loss = (is_weights * (expected_Q.detach() - curr_Q) ** 2).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA