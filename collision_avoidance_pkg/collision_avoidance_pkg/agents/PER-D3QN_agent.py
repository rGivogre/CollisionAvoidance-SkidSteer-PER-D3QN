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
TAU = 0.005                 # Soft update del target network (stabile e progressivo)

# PER Specific Hyperparameters
PER_ALPHA = 0.6       # Determina il livello di prioritizzazione utilizzato
PER_BETA_START = 0.4  # Valore iniziale del fattore di correzione Importance Sampling (IS)
PER_BETA_STEPS = 100000 # Numero di step per far flettere linearmente beta fino a 1.0
PER_EPSILON = 1e-5    # Piccola costante positiva per garantire priorità non nulle


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


class PERD3QNAgent:
    def __init__(self, learning_rate=LEARNING_RATE):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        # Inizializzazione Policy e Target Network con Architettura Dueling
        self.policy_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        self.target_net = DuelingQNetwork(STATE_SIZE, ACTION_SIZE).to(self.train_device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())   
        self.target_net.eval()  # Il target network serve solo per inferenza statica
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
        # Setup del Prioritized Replay Buffer tramite cpprb
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
        
        # Parametri Epsilon-greedy
        self.epsilon = 1.0
        self.step_count = 0
        
        # Tracciamento e annealing di Beta per l'Importance Sampling
        self.beta = PER_BETA_START

    def get_action(self, state):
        """ Seleziona un'azione usando una politica epsilon-greedy. """
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)    
        
        return torch.argmax(q_values).item()    

    def store_transition(self, state, action, reward, next_state, done):
        """Memoria delle transizioni configurata per l'ambiente dizionario di cpprb."""
        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def train_step(self):
        """ Estrae un mini-batch tramite PER, calcola la Huber Loss pesata e aggiorna la rete. """
        if self.memory.get_stored_size() < BATCH_SIZE:   
            return
            
        # Calcolo dell'annealing lineare di beta verso 1.0
        self.beta = min(1.0, PER_BETA_START + self.step_count * (1.0 - PER_BETA_START) / PER_BETA_STEPS)

        # Campionamento dal buffer prioritizzato
        samples = self.memory.sample(BATCH_SIZE, self.beta)
        
        states = torch.FloatTensor(samples["state"]).to(self.train_device)
        actions = torch.LongTensor(samples["action"]).to(self.train_device)
        rewards = torch.FloatTensor(samples["reward"]).to(self.train_device)
        next_states = torch.FloatTensor(samples["next_state"]).to(self.train_device)
        dones = torch.FloatTensor(samples["done"]).to(self.train_device)
        
        is_weights = torch.FloatTensor(samples["weights"]).unsqueeze(1).to(self.train_device)
        idxs = samples["indexes"]

        # Estrazione dei valori Q correnti dal Dueling Policy Network
        curr_Q = self.policy_net(states).gather(1, actions)
        
        # Logica Double DQN: l'azione migliore viene dalla Policy Net, la valutazione dalla Target Net
        best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_Q = self.target_net(next_states).gather(1, best_next_actions) 
        
        # Calcolo dei target Q-value attesi
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))

        # Calcolo degli errori TD assoluti per l'aggiornamento delle priorità nell'albero di cpprb
        with torch.no_grad():
            td_errors = (expected_Q - curr_Q).cpu().numpy().flatten()
            
        # Aggiornamento delle priorità nel buffer (evitando valori a zero assoluto)
        self.memory.update_priorities(idxs, np.abs(td_errors) + PER_EPSILON)

        # Calcolo della Smooth L1 Loss (Huber) pesata elemento per elemento con l'Importance Sampling
        elementwise_loss = nn.SmoothL1Loss(reduction='none')(curr_Q, expected_Q.detach())
        loss = (is_weights * elementwise_loss).mean()

        # Ottimizzazione e Gradient Clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        
        # Aggiornamento progressivo Soft Update (TAU) del Target Network ad ogni step di addestramento
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def update_epsilon(self):
        """ Decadimento controllato di epsilon al termine di ogni episodio. """
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_RATE_BETA
            if self.epsilon < MIN_EPSILON:
                self.epsilon = MIN_EPSILON