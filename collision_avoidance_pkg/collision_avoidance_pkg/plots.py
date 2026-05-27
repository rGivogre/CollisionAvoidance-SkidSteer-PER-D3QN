import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    plot_data_dir = os.path.join(project_root, 'plot_data')
    
    try:
        rewards = np.load(os.path.join(plot_data_dir, 'rewards.npy'))
        actions = np.load(os.path.join(plot_data_dir, 'actions.npy'))
        epsilons = np.load(os.path.join(plot_data_dir, 'epsilons.npy'))
        return rewards, actions, epsilons, plot_data_dir
    except FileNotFoundError:
        print("Error: Log files not found. Make sure training has run past at least 100 episodes.")
        return None, None, None, None

def load_crash_data():
    try:
        crashes = np.load('logs/crashes.npy')
        return crashes
    except FileNotFoundError:
        print("Warning: 'logs/crashes.npy' not found. Skip crash heatmap or wait for a crash to occur.")
        return None

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
    plt.savefig(os.path.join(save_dir, 'convergence_plot.png'), dpi=300)
    plt.show()

def plot_action_heatmap(actions, save_dir, blocks_of_episodes=200):
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
    plt.savefig(os.path.join(save_dir, 'action_heatmap.png'), dpi=300)
    plt.show()

def plot_exploration_vs_exploitation(rewards, epsilons, save_dir):
    """Plots reward against the actual epsilon decay recorded during training."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon (Exploration)', color=color)
    # Using the true recorded epsilon history prevents any simulation biases
    ax1.plot(epsilons, color=color, linestyle='--', linewidth=2, label='Actual Epsilon Decay')
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
    plt.savefig(os.path.join(save_dir, 'exploration_vs_reward.png'), dpi=300)
    plt.show()

def plot_crashes_heatmap(crashes):
    """Overlays a 2D density heatmap of collision points onto the world map image."""
    if crashes is None or len(crashes) == 0:
        return

    try:
        map_img = plt.imread('logs/map_screenshot.png')
    except FileNotFoundError:
        print("Error: To generate the overlay, save your map screenshot as 'logs/map_screenshot.png'")
        return

    x_crashes = crashes[:, 0]
    y_crashes = crashes[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Definizione dei confini fisici del mondo di Gazebo in metri [xmin, xmax, ymin, ymax]
    # Modifica questi parametri in base alle dimensioni effettive del tuo file .world
    world_extent = [-10.0, 10.0, -10.0, 10.0]

    # Mostra l'immagine di sfondo mappandola sulle coordinate reali di Gazebo
    ax.imshow(map_img, extent=world_extent, origin='upper', alpha=0.8)

    # Disegna il density heatmap (KDE) se ci sono abbastanza campioni per calcolare la densità
    if len(crashes) > 5:
        sns.kdeplot(
            x=x_crashes, y=y_crashes, 
            ax=ax,
            fill=True, 
            cmap="Reds", 
            alpha=0.5,      # Trasparenza per visualizzare i muri sottostanti
            thresh=0.1,     # Esclude il rumore a bassissima densità
            levels=10       # Dettaglio dei livelli di densità
        )
    
    # Disegna i singoli punti esatti di impatto come piccoli punti scuri
    ax.scatter(x_crashes, y_crashes, color='darkred', s=6, alpha=0.6, label='Crash Site')

    ax.set_title('Robot Collision Distribution Heatmap Overlay')
    ax.set_xlabel('Global X Position (meters)')
    ax.set_ylabel('Global Y Position (meters)')
    ax.set_xlim(world_extent[0], world_extent[1])
    ax.set_ylim(world_extent[2], world_extent[3])
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    plt.savefig('logs/crash_heatmap_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_crashes_heatmap(crashes):
    """Overlays a 2D density heatmap of collision points onto the world map image."""
    if crashes is None or len(crashes) == 0:
        return

    try:
        map_img = plt.imread('logs/map_screenshot.png')
    except FileNotFoundError:
        print("Error: To generate the overlay, save your map screenshot as 'logs/map_screenshot.png'")
        return

    x_crashes = crashes[:, 0]
    y_crashes = crashes[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Definizione dei confini fisici del mondo di Gazebo in metri [xmin, xmax, ymin, ymax]
    # Modifica questi parametri in base alle dimensioni effettive del tuo file .world
    world_extent = [-10.0, 10.0, -10.0, 10.0]

    # Mostra l'immagine di sfondo mappandola sulle coordinate reali di Gazebo
    ax.imshow(map_img, extent=world_extent, origin='upper', alpha=0.8)

    # Disegna il density heatmap (KDE) se ci sono abbastanza campioni per calcolare la densità
    if len(crashes) > 5:
        sns.kdeplot(
            x=x_crashes, y=y_crashes, 
            ax=ax,
            fill=True, 
            cmap="Reds", 
            alpha=0.5,      # Trasparenza per visualizzare i muri sottostanti
            thresh=0.1,     # Esclude il rumore a bassissima densità
            levels=10       # Dettaglio dei livelli di densità
        )
    
    # Disegna i singoli punti esatti di impatto come piccoli punti scuri
    ax.scatter(x_crashes, y_crashes, color='darkred', s=6, alpha=0.6, label='Crash Site')

    ax.set_title('Robot Collision Distribution Heatmap Overlay')
    ax.set_xlabel('Global X Position (meters)')
    ax.set_ylabel('Global Y Position (meters)')
    ax.set_xlim(world_extent[0], world_extent[1])
    ax.set_ylim(world_extent[2], world_extent[3])
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    plt.savefig('logs/crash_heatmap_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    rewards, actions, epsilons, plot_data_dir = load_data()
    crashes = load_crash_data()
    
    if rewards is not None:
        plot_convergence(rewards, plot_data_dir)
        plot_action_heatmap(actions, plot_data_dir, blocks_of_episodes=100)
        plot_exploration_vs_exploitation(rewards, epsilons, plot_data_dir)
    if crashes is not None:
        plot_crashes_heatmap(crashes)