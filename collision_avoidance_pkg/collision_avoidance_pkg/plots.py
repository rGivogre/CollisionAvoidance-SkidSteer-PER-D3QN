import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(run_folder=None):
    base_plot_dir = 'plot_data'
    
    if run_folder:
        plot_data_dir = os.path.join(base_plot_dir, run_folder)
    else:
        # Sort folders by modification time and pick the most recent one
        try:
            folders = [os.path.join(base_plot_dir, d) for d in os.listdir(base_plot_dir) if os.path.isdir(os.path.join(base_plot_dir, d))]
            if not folders:
                raise ValueError
            plot_data_dir = max(folders, key=os.path.getmtime)
            print(f"Auto-selected latest run: {os.path.basename(plot_data_dir)}")
        except ValueError:
            print("Error: No training runs found in the plot_data directory.")
            return None, None, None, None

    try:
        rewards = np.load(os.path.join(plot_data_dir, 'rewards.npy'))
        actions = np.load(os.path.join(plot_data_dir, 'actions.npy'))
        epsilons = np.load(os.path.join(plot_data_dir, 'epsilons.npy'))
        return rewards, actions, epsilons, plot_data_dir
    except FileNotFoundError as e:
        print(f"Error: Log files not found. {e}")
        return None, None, None, None

def load_crash_data(plot_data_dir):
    try:
        crashes = np.load(os.path.join(plot_data_dir, 'crashes.npy'))
        return crashes
    except FileNotFoundError:
        print("Warning: 'crashes.npy' not found. Skip crash heatmap or wait for a crash to occur.")
        return None

def plot_convergence(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color='royalblue', alpha=0.3, label='Raw Episode Reward')
    
    window_size = 500
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='crimson', linewidth=2, label='Moving Avg (500 Ep)')
        
    plt.title('DDQN Learning Convergence Curve')
    plt.xlabel('Training Epochs (Episodes)')
    plt.ylabel('Accumulated Rewards')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'convergence_plot.png'), dpi=300)
    plt.show()

def plot_action_heatmap(actions, save_dir, blocks_of_episodes=200):
    valid_episodes = [ep[ep != -1] for ep in actions]
    num_episodes = len(valid_episodes)
    num_blocks = num_episodes // blocks_of_episodes
    
    if num_blocks == 0:
        print("Not enough data yet to compute an action heatmap blocks.")
        return

    heatmap_matrix = np.zeros((11, num_blocks))
    for block in range(num_blocks):
        start_ep = block * blocks_of_episodes
        end_ep = start_ep + blocks_of_episodes
        block_actions = np.concatenate(valid_episodes[start_ep:end_ep])
        counts = np.bincount(block_actions.astype(int), minlength=11)
        heatmap_matrix[:, block] = counts / len(block_actions)

    plt.figure(figsize=(12, 6))
    y_labels = [f"A${m}: {round(-0.8 + 0.16*m, 2)}" for m in range(11)]
    x_labels = [f"{i*blocks_of_episodes}-{(i+1)*blocks_of_episodes}" for i in range(num_blocks)]
    
    sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                yticklabels=y_labels, xticklabels=x_labels)
    
    plt.title('Steering Action Selection Probability Heatmap Over Time')
    plt.ylabel('Actions (Index & Angular Velocity rad/s)')
    plt.xlabel('Episode Windows')
    plt.savefig(os.path.join(save_dir, 'action_heatmap.png'), dpi=300)
    plt.show()

def plot_exploration_vs_exploitation(rewards, epsilons, save_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon (Exploration)', color=color)
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

def plot_crashes_heatmap(crashes, save_dir):
    if crashes is None or len(crashes) == 0:
        return

    map_img_path = os.path.join('images', 'map_screenshot.png')

    try:
        map_img = plt.imread(map_img_path)
    except FileNotFoundError:
        print(f"Error: To generate the overlay, save your map screenshot as '{map_img_path}'")
        return

    x_crashes = crashes[:, 0]
    y_crashes = crashes[:, 1]
    fig, ax = plt.subplots(figsize=(10, 10))

    world_extent = [-10.0, 10.0, -10.0, 10.0]
    ax.imshow(map_img, extent=world_extent, origin='upper', alpha=0.8)

    if len(crashes) > 5:
        sns.kdeplot(
            x=x_crashes, y=y_crashes, 
            ax=ax,
            fill=True, 
            cmap="Reds", 
            alpha=0.5,      
            thresh=0.1,     
            levels=10       
        )
    
    ax.scatter(x_crashes, y_crashes, color='darkred', s=6, alpha=0.6, label='Crash Site')

    ax.set_title('Robot Collision Distribution Heatmap Overlay')
    ax.set_xlabel('Global X Position (meters)')
    ax.set_ylabel('Global Y Position (meters)')
    ax.set_xlim(world_extent[0], world_extent[1])
    ax.set_ylim(world_extent[2], world_extent[3])
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    plt.savefig(os.path.join(save_dir, 'crash_heatmap_overlay.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot DDQN training data.")
    parser.add_argument('--run', type=str, default=None, help="Name of the timestamp folder(e.g., '20260528_153022'). If omitted, loads the latest.")
    args = parser.parse_args()

    rewards, actions, epsilons, plot_data_dir = load_data(args.run)
    
    if rewards is not None:
        crashes = load_crash_data(plot_data_dir)
        
        plot_convergence(rewards, plot_data_dir)
        plot_action_heatmap(actions, plot_data_dir, blocks_of_episodes=100)
        plot_exploration_vs_exploitation(rewards, epsilons, plot_data_dir)
        if crashes is not None:
            plot_crashes_heatmap(crashes, plot_data_dir)
