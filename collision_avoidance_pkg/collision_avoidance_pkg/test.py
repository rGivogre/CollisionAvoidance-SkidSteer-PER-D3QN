import rclpy
import torch
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from .gazebo_env import GazeboEnv
from .train import MAX_STEPS_PER_EPISODE
from .agents.DDQN_agent import QNetwork, STATE_SIZE, ACTION_SIZE
from .agents.D3QN_agent import DuelingQNetwork

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

def get_network_architecture(model_dir: str) -> tuple[torch.nn.Module, str]:
    """Reads runconfig.json to dynamically load the correct neural network structure."""
    config_path = os.path.join(model_dir, 'runconfig.json')
    
    if not os.path.exists(config_path):
        print("[WARNING] runconfig.json not found. Defaulting to Standard QNetwork.")
        return QNetwork(STATE_SIZE, ACTION_SIZE), "Unknown-DDQN"

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    agent_type = config.get('agent_key', 'DDQN')
    run_name = config.get('run_name', 'unnamed_run')
    
    if 'D3QN' in agent_type:
        print(f"[TEST] Architecture detected: Dueling Q-Network ({agent_type})")
        return DuelingQNetwork(STATE_SIZE, ACTION_SIZE), run_name
    else:
        print(f"[TEST] Architecture detected: Standard Q-Network ({agent_type})")
        return QNetwork(STATE_SIZE, ACTION_SIZE), run_name

def main():
    parser = argparse.ArgumentParser(description="Skidbot Universal Testing Node")
    parser.add_argument('--model_path', type=str, default='', help='Path to the .pth model or directory')
    parser.add_argument('--speed', type=float, default=0.3, help='Linear velocity for the robot')
    parser.add_argument('--map_name', type=str, default='hardTest', help='Map name used to select spawn zones')
    parser.add_argument('--episodes', type=int, default=30, help='Number of test episodes for statistical significance')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='Max steps per episode')
    parser.add_argument('--lock_step', type=bool, default=True, help='Enable physics lock-step during inference')
    
    args, _ = parser.parse_known_args()
    
    resolved_model_path = resolve_model_path(args.model_path)
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        print("[TEST] Error: No valid model checkpoint was found.")
        return

    model_dir = os.path.dirname(resolved_model_path)
    
    # Initialize ROS 2 and Gazebo
    rclpy.init()
    env = GazeboEnv(linear_speed=args.speed, lock_step=args.lock_step, map_name=args.map_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[TEST] Using device: {device}")
    print(f"[TEST] Loading model from: {resolved_model_path}")
    
    # Dynamically load the correct architecture
    policy_net, run_name = get_network_architecture(model_dir)
    policy_net = policy_net.to(device)
    
    # Load weights
    checkpoint_data = torch.load(resolved_model_path, map_location=device)
    if isinstance(checkpoint_data, dict) and 'model_state' in checkpoint_data:
        policy_net.load_state_dict(checkpoint_data['model_state'])
    else:
        policy_net.load_state_dict(checkpoint_data)
        
    policy_net.eval()

    print(f"\n--- Starting Evaluation: {run_name} ({args.episodes} Episodes) ---")
    
    results = []

    try:
        for episode in range(1, args.episodes + 1):
            state, start_coords = env.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(args.max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                
                # Pure exploitation during test phase
                action_idx = torch.argmax(q_values).item()
                next_state, reward, done, crash_coords = env.step(action_idx)
                
                state = next_state
                episode_reward += float(reward)
                steps += 1
                
                if done:
                    break
            
            # Log episode statistics
            collision_occurred = bool(done)
            results.append({
                "Episode": episode,
                "Reward": episode_reward,
                "Steps": steps,
                "Collision": collision_occurred,
                "Map": args.map_name,
                "Linear_Speed": args.speed
            })
            
            status = "COLLISION" if collision_occurred else "SURVIVED"
            print(f"Ep {episode:02d} | Status: {status:9} | Steps: {steps:4d} | Reward: {episode_reward:7.2f}")

    except (KeyboardInterrupt, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "Unable to convert call argument" not in str(e):
            raise e
        print("\n[TEST] Evaluation interrupted safely.")

    finally:
        # Calculate Metrics
        df = pd.DataFrame(results)
        if not df.empty:
            mean_reward = df["Reward"].mean()
            std_reward = df["Reward"].std()
            survival_rate = (1.0 - df["Collision"].mean()) * 100
            
            print("\n" + "="*50)
            print(f"FINAL EVALUATION METRICS: {run_name}")
            print("="*50)
            print(f"Mean Reward:        {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Survival Rate:      {survival_rate:.1f}%")
            print(f"Avg Episode Length: {df['Steps'].mean():.1f} steps")
            print("="*50)

            # Export to CSV for plotting
            results_dir = os.path.join("test_results")
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"eval_{run_name}_{args.map_name}_{timestamp}.csv"
            csv_path = os.path.join(results_dir, csv_filename)
            
            df.to_csv(csv_path, index=False)
            print(f"\n[TEST] Detailed results saved to: {csv_path}")
            
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()