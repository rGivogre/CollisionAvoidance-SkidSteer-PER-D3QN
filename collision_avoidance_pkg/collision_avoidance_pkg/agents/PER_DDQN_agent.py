import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from cpprb import PrioritizedReplayBuffer  

# Hyperparameters from the paper and standard RL practices
STATE_SIZE = 50
ACTION_SIZE = 11
HIDDEN_NEURONS = 300
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 2.5e-4
MIN_EPSILON = 0.05
DECAY_RATE_BETA = 0.999  
TAU = 0.005                 # MIGLIORIA: per soft update del target network (preso da DDQN)

# PER Specific Hyperparameters
PER_ALPHA = 0.6       # Determines how much prioritization is used
PER_BETA_START = 0.4  # Importance sampling correction factor starting value
PER_BETA_STEPS = 100000 # Number of steps to anneal beta to 1.0
PER_EPSILON = 1e-5    # Small positive constant to ensure non-zero priority


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


class PERDDQNAgent:
    def __init__(self, learning_rate=LEARNING_RATE): # MIGLIORIA: accetta il parametro lr come in DDQN
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        # Initialize Policy Network and Target Network
        self.policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   
        self.target_net.eval()  # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
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
            # Exploration: choose random action
            return random.randrange(ACTION_SIZE)
            
        # Exploitation: 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)    
        
        return torch.argmax(q_values).item()    

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the replay memory."""
        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def train_step(self):
        """ Samples a mini-batch using cpprb and optimizes the network. """
        if self.memory.get_stored_size() < BATCH_SIZE:   
            return
            
        # Anneal beta
        self.beta = min(1.0, PER_BETA_START + self.step_count * (1.0 - PER_BETA_START) / PER_BETA_STEPS)

        # Sampling from PER
        samples = self.memory.sample(BATCH_SIZE, self.beta)
        
        states = torch.FloatTensor(samples["state"]).to(self.train_device)
        actions = torch.LongTensor(samples["action"]).to(self.train_device)
        rewards = torch.FloatTensor(samples["reward"]).to(self.train_device)
        next_states = torch.FloatTensor(samples["next_state"]).to(self.train_device)
        dones = torch.FloatTensor(samples["done"]).to(self.train_device)
        
        is_weights = torch.FloatTensor(samples["weights"]).unsqueeze(1).to(self.train_device)
        idxs = samples["indexes"]

        # Get current Q-values from Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # DDQN logic: Select best action from Policy Network, evaluate it using Target Network
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions)
        
        # Calculate target Q-values 
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # Calculate individual absolute errors (TD-errors) to update priorities
        with torch.no_grad():
            td_errors = (expected_Q - curr_Q).cpu().numpy().flatten()
            
        # Update tree priorities in cpprb
        self.memory.update_priorities(idxs, np.abs(td_errors) + PER_EPSILON)

        elementwise_loss = nn.SmoothL1Loss(reduction='none')(curr_Q, expected_Q.detach())
        loss = (is_weights * elementwise_loss).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        self.step_count += 1
        
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