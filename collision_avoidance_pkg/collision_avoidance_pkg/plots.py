import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    try:
        rewards = np.load('logs/rewards.npy')
        actions = np.load('logs/actions.npy')
        return rewards, actions
    except FileNotFoundError:
        print("Error: Log files not found. Make sure training has run past at least 100 episodes.")
        return None, None

def plot_convergence(rewards):
    """Plots the raw rewards per episode and a moving average curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color='royalblue', alpha=0.3, label='Raw Episode Reward')
    
    # Calculate a rolling window average (size 100)
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='crimson', linewidth=2, label='Moving Avg (100 Ep)')
        
    plt.title('DQN Learning Convergence Curve')
    plt.xlabel('Training Epochs (Episodes)')
    plt.ylabel('Accumulated Rewards')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('logs/convergence_plot.png', dpi=300)
    plt.show()

def plot_action_heatmap(actions, blocks_of_episodes=200):
    """
    Generates a heatmap showing action distributions over time chunks.
    Helps show if the agent progresses from chaotic steering to stable actions.
    """
    # Filter out padding values (-1) used to shape arrays in train.py
    valid_episodes = [ep[ep != -1] for ep in actions]
    
    num_episodes = len(valid_episodes)
    num_blocks = num_episodes // blocks_of_episodes
    
    if num_blocks == 0:
        print("Not enough data yet to compute an action heatmap blocks.")
        return

    # Action indices range from 0 to 10 (11 total discrete actions)
    heatmap_matrix = np.zeros((11, num_blocks))
    
    for block in range(num_blocks):
        start_ep = block * blocks_of_episodes
        end_ep = start_ep + blocks_of_episodes
        
        # Gather all actions executed in this block
        block_actions = np.concatenate(valid_episodes[start_ep:end_ep])
        
        # Calculate frequency distribution
        counts = np.bincount(block_actions.astype(int), minlength=11)
        heatmap_matrix[:, block] = counts / len(block_actions)

    plt.figure(figsize=(12, 6))
    # Y-axis labels mapped to the paper's angular velocity values
    y_labels = [f"A{m}: {round(-0.8 + 0.16*m, 2)}" for m in range(11)]
    x_labels = [f"{i*blocks_of_episodes}-{(i+1)*blocks_of_episodes}" for i in range(num_blocks)]
    
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                yticklabels=y_labels, xticklabels=x_labels)
    
    plt.title('Steering Action Selection Probability Heatmap Over Time')
    plt.ylabel('Actions (Index & Angular Velocity rad/s)')
    plt.xlabel('Episode Windows')
    plt.savefig('logs/action_heatmap.png', dpi=300)
    plt.show()

def plot_exploration_vs_exploitation(rewards):
    """Plots reward against an estimation of the historical epsilon value decay."""
    episodes = len(rewards)
    epsilon = 1.0
    epsilon_history = []
    
    # Reconstruct epsilon curve matching training decay
    for _ in range(episodes):
        epsilon_history.append(epsilon)
        if epsilon > 0.05:
            epsilon *= 0.999

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon (Exploration)', color=color)
    ax1.plot(epsilon_history, color=color, linestyle='--', linewidth=2, label='Epsilon Value')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Rewards', color=color)
    
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(rewards)), moving_avg, color=color, label='Moving Avg Reward')
    else:
        ax2.plot(rewards, color=color, label='Raw Reward')
        
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Exploration vs Exploitation Progress Transition')
    fig.tight_layout()
    plt.savefig('logs/exploration_vs_reward.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    rewards, actions = load_data()
    if rewards is not None:
        plot_convergence(rewards)
        plot_action_heatmap(actions, blocks_of_episodes=100) # Adjust block chunk sizes if needed
        plot_exploration_vs_exploitation(rewards)