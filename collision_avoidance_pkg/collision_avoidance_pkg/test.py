import rclpy
import torch
import os
import argparse
import numpy as np
from pathlib import Path
from .gazebo_env import GazeboEnv
from .train import MAX_STEPS_PER_EPISODE
from .agents.DDQN_agent import QNetwork, STATE_SIZE, ACTION_SIZE

def resolve_model_path(model_path_hint: str) -> str | None:
    """Resolve a usable model path from an explicit file, a directory, or the latest run."""
    candidate = Path(model_path_hint).expanduser()

    if candidate.is_file():
        return str(candidate)

    if candidate.is_dir():
        checkpoint_files = sorted(candidate.rglob('*.pth'), key=lambda path: path.stat().st_mtime)
        if checkpoint_files:
            return str(checkpoint_files[-1])

    models_dir = Path('models')
    if models_dir.exists():
        checkpoint_files = sorted(models_dir.rglob('*.pth'), key=lambda path: path.stat().st_mtime)
        if checkpoint_files:
            return str(checkpoint_files[-1])

    return None

def main():
    parser = argparse.ArgumentParser(description="Skidbot DDQN Testing Node")
    parser.add_argument('--model_path', type=str, default='', help='Path to the .pth model or directory')
    parser.add_argument('--speed', type=float, default=0.3, help='Linear velocity for the robot')
    parser.add_argument('--map_name', type=str, default='map2', help='Map name used to select spawn zones')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes to run')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='Max steps per episode')
    parser.add_argument('--lock_step', type=bool, default=False, help='Enable physics lock-step')
    
    args, _ = parser.parse_known_args()
    
    resolved_model_path = resolve_model_path(args.model_path)

    # Initialize ROS 2, Gazebo environment and load the trained model
    rclpy.init()
    env = GazeboEnv(linear_speed=args.speed, lock_step=args.lock_step, map_name=args.map_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TEST] Using device: {device}")
    print(f"[TEST] Linear speed: {args.speed}")
    print(f"[TEST] Lock-step enabled: {args.lock_step}")
    print(f"[TEST] Total episodes: {args.episodes}")
    
    # Configure the neural network architecture
    policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
    
    # Load weights
    if resolved_model_path and os.path.exists(resolved_model_path):
        print(f"[TEST] Loading model from: {resolved_model_path}")
        checkpoint_data = torch.load(resolved_model_path, map_location=device)
        if isinstance(checkpoint_data, dict) and 'model_state' in checkpoint_data:
            policy_net.load_state_dict(checkpoint_data['model_state'])
        else:
            policy_net.load_state_dict(checkpoint_data)
        policy_net.eval()
    else:
        print("[TEST] Error: No valid model checkpoint was found.")
        env.destroy_node()
        rclpy.shutdown()
        return

    print("\n--- Starting Skidbot Test Phase ---")
    collision_count = 0
    episode_rewards = []

    for episode in range(1, args.episodes + 1):
        state, start_coords = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nStarting Test Episode {episode}...")
        for step in range(args.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            next_state, reward, done, crash_coords = env.step(action_idx)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                collision_count += 1
                print(f"Episode terminated by collision after {steps} steps.")
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode} Result -> Steps completed: {steps}, Total Reward: {episode_reward:.2f}")

    average_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    print(f"\nTest phase completed. Average reward: {average_reward:.2f}, Collisions: {collision_count}/{args.episodes}")
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()