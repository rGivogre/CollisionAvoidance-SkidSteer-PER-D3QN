import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

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
PER_ALPHA = 0.6       # Determines how much prioritization is used (0 = uniform, 1 = full prioritization)
PER_BETA_START = 0.4  # Importance sampling correction factor starting value
PER_BETA_STEPS = 100000 # Number of steps to anneal beta to 1.0
PER_EPSILON = 1e-5    # Small positive constant to ensure zero-error experiences can still be sampled


class SumTree:
    """ A binary tree data structure where parent nodes are the sum of children nodes. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def update(self, idx, p):
        """ Update priority score in the tree. """
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, p, data):
        """ Add a new experience with a priority score to the tree. """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        """ Sample an experience based on a priority value v. """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """ Replay Buffer backed by a SumTree to enable Prioritized Experience Replay. """
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def store(self, transition):
        # New transitions get maximum priority so they are guaranteed to be trained on at least once
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, transition)

    def sample(self, batch_size, beta):
        mini_batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            
            priorities.append(p)
            idxs.append(idx)
            mini_batch.append(data)

        # Calculate Importance Sampling Weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        weights /= weights.max() # Normalize weights
        
        return mini_batch, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + PER_EPSILON) ** PER_ALPHA
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


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
    def __init__(self):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Policy Network and Target Network
        self.policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   # Copy weights from policy network to target network
        self.target_net.eval()  # Target network is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    
        # CHANGED: Replaced standard deque with PrioritizedReplayBuffer
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        
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
            q_values = self.policy_net(state_tensor)    # get Q-values for all possible actions
        
        return torch.argmax(q_values).item()    # return index of the action with the highest Q-value

    def store_transition(self, state, action, reward, next_state, done):
        """Stores the transition in the replay memory."""
        # CHANGED: Uses .store() interface tailored for PER
        self.memory.store((state, action, reward, next_state, done))

    def train_step(self):
        """
        Samples a mini-batch from memory based on priority and performs a 
        gradient descent step using Importance Sampling Weights.
        """
        if len(self.memory) < BATCH_SIZE:   # if we don't have enough samples in memory, skip training
            return
            
        # CHANGED: Incrementally anneal beta towards 1.0
        self.beta = min(1.0, PER_BETA_START + self.step_count * (1.0 - PER_BETA_START) / PER_BETA_STEPS)

        # CHANGED: Sample mini-batch alongside internal tree indices and Importance Sampling weights
        minibatch, idxs, weights = self.memory.sample(BATCH_SIZE, self.beta)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.train_device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.train_device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.train_device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.train_device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.train_device)
        is_weights = torch.FloatTensor(weights).unsqueeze(1).to(self.train_device)

        # Get current Q-values from Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # DDQN logic: Select best action from Policy Network, evaluate it using Target Network
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions)
        
        # Calculate target Q-values 
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # CHANGED: Calculate individual absolute errors (TD-errors) to update priorities in the tree
        with torch.no_grad():
            td_errors = (expected_Q - curr_Q).cpu().numpy().flatten()
        self.memory.update_priorities(idxs, td_errors)

        # CHANGED: Apply Importance Sampling weights to the elements of the loss function
        # This counteracts the bias brought by non-uniform sampling
        loss = (is_weights * (expected_Q.detach() - curr_Q) ** 2).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        
        # Replace target network parameters every N steps
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Decays epsilon by beta. To be called at the end of each epoch.
        """
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA