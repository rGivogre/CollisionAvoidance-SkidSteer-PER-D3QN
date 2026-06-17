import argparse
import json
import re
import pickle
from pathlib import Path

import numpy as np
import torch
import rclpy

from .gazebo_env import GazeboEnv
from .agents.DDQN_agent import DDQNAgent, MIN_EPSILON, DECAY_RATE_BETA
from .train import MAX_EPISODES, MAX_STEPS_PER_EPISODE, SAVE_EVERY


def resolve_checkpoint(model_path_hint: str) -> Path | None:
    """
    Resolve a checkpoint file from a path hint.
    - If model_path_hint is empty, search models/ for the most recent checkpoint
    - If it's a file, use it directly
    - If it's a directory, find the latest checkpoint in it
    """
    if not model_path_hint or model_path_hint == '':
        # No path provided, search models/ for the most recent checkpoint
        models_root = Path('models')
        if models_root.exists():
            checkpoint_files = sorted(models_root.rglob('*.pth'), key=lambda path: path.stat().st_mtime)
            if checkpoint_files:
                latest = checkpoint_files[-1]
                print(f"No model path specified. Using most recent: {latest}")
                return latest
        return None

    candidate = Path(model_path_hint).expanduser()

    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        checkpoint_files = sorted(candidate.rglob('*.pth'), key=lambda path: path.stat().st_mtime)
        if checkpoint_files:
            return checkpoint_files[-1]

    # If provided path doesn't exist, try searching models/ as fallback
    models_root = Path('models')
    if models_root.exists():
        checkpoint_files = sorted(models_root.rglob('*.pth'), key=lambda path: path.stat().st_mtime)
        if checkpoint_files:
            return checkpoint_files[-1]

    return None


def parse_episode_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract episode number from checkpoint filename (e.g., ddqn_ep1000.pth -> 1000)."""
    match = re.search(r'ep(\d+)', checkpoint_path.name)
    return int(match.group(1)) if match else 0


def load_run_config(run_folder: Path) -> dict | None:
    """Load runconfig.json from a run folder."""
    config_path = run_folder / 'runconfig.json'
    if config_path.exists():
        with open(config_path, 'r') as fp:
            return json.load(fp)
    return None


def estimate_epsilon_from_episode(episode: int) -> float:
    """Estimate epsilon decay from episode number."""
    if episode <= 0:
        return 1.0
    epsilon = 1.0 * (DECAY_RATE_BETA ** episode)
    return max(epsilon, MIN_EPSILON)


def load_array(file_path: Path, allow_pickle: bool = False) -> np.ndarray | None:
    """Load numpy array if it exists."""
    if file_path.exists():
        return np.load(file_path, allow_pickle=allow_pickle)
    return None


def append_and_save(file_path: Path, existing_data, new_data) -> np.ndarray:
    """Append new data to existing array and save."""
    new_array = np.array(new_data)
    if existing_data is None or existing_data.size == 0:
        combined = new_array
    else:
        combined = np.concatenate([existing_data, new_array], axis=0)
    np.save(file_path, combined)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Resume Skidbot DDQN training from a saved checkpoint. "
                    "Run from the CollisionAvoidance-RL folder. Paths in runconfig.json should be relative."
    )
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to a .pth checkpoint or directory containing checkpoints. '
                             'If omitted, uses the most recent checkpoint from models/')
    parser.add_argument('--speed', type=float, default=None,
                        help='Override linear velocity (from runconfig if omitted)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate (from runconfig if omitted)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override max steps per episode (from runconfig if omitted)')
    parser.add_argument('--target_episodes', type=int, default=MAX_EPISODES,
                        help='Episode count to resume training until')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Override save frequency (from runconfig if omitted)')
    parser.add_argument('--start_epsilon', type=float, default=None,
                        help='Optional epsilon to start from (estimated from checkpoint episode if omitted)')

    args = parser.parse_args()

    # Resolve checkpoint file
    resolved_checkpoint = resolve_checkpoint(args.model_path)
    if resolved_checkpoint is None:
        print(f"Error: could not resolve checkpoint from '{args.model_path}'.")
        return

    # Load run configuration
    run_config = load_run_config(resolved_checkpoint.parent)
    if run_config is None:
        print(f"Error: runconfig.json not found in '{resolved_checkpoint.parent}'.")
        print("Cannot restore without runconfig.json. Please provide one in the model folder.")
        return

    print(f"Loaded runconfig from: {resolved_checkpoint.parent / 'runconfig.json'}")

    # Resolve plot_data directory using runconfig
    plot_data_dir_key = run_config.get('plot_data_dir')
    if plot_data_dir_key:
        plot_data_dir = Path(plot_data_dir_key)  # Should be relative to CollisionAvoidance-RL
    else:
        # Fallback: use run folder name
        run_folder_name = resolved_checkpoint.parent.name
        plot_data_dir = Path('plot_data') / run_folder_name

    if not plot_data_dir.exists():
        print(f"Error: plot data folder not found at '{plot_data_dir}'.")
        return

    # Prepare data file paths
    rewards_path = plot_data_dir / 'rewards.npy'
    actions_path = plot_data_dir / 'actions.npy'
    crashes_path = plot_data_dir / 'crashes.npy'
    epsilons_path = plot_data_dir / 'epsilons.npy'

    # Load existing training data
    existing_rewards = load_array(rewards_path)
    existing_actions = load_array(actions_path, allow_pickle=True)
    existing_crashes = load_array(crashes_path, allow_pickle=True)
    existing_epsilons = load_array(epsilons_path)

    # Extract episode number from checkpoint
    current_episode = parse_episode_from_checkpoint(resolved_checkpoint)
    if current_episode <= 0:
        print(f"Warning: Could not parse episode from checkpoint '{resolved_checkpoint.name}'. Assuming ep 0.")

    if args.target_episodes <= current_episode:
        print(f"Checkpoint is already at episode {current_episode}. Nothing to resume to {args.target_episodes}.")
        return

    # Resolve hyperparameters: CLI args override runconfig, which provides defaults
    speed = args.speed if args.speed is not None else run_config.get('speed', 0.3)
    learning_rate = args.learning_rate if args.learning_rate is not None else run_config.get('learning_rate', 2.5e-4)
    max_steps_per_episode = args.max_steps if args.max_steps is not None else run_config.get('max_steps_per_episode', MAX_STEPS_PER_EPISODE)
    save_every = args.save_every if args.save_every is not None else run_config.get('save_every', SAVE_EVERY)

    resume_epsilon = args.start_epsilon if args.start_epsilon is not None else estimate_epsilon_from_episode(current_episode)
    resume_epsilon = min(max(resume_epsilon, MIN_EPSILON), 1.0)

    # Print configuration
    print(f"\n{'='*70}")
    print(f"Resuming Training Session")
    print(f"{'='*70}")
    print(f"Checkpoint:        {resolved_checkpoint}")
    print(f"Run folder:        {resolved_checkpoint.parent.name}")
    print(f"Plot data:         {plot_data_dir}")
    print(f"Current episode:   {current_episode}")
    print(f"Target episode:    {args.target_episodes}")
    print(f"Speed:             {speed}")
    print(f"Learning rate:     {learning_rate}")
    print(f"Max steps/ep:      {max_steps_per_episode}")
    print(f"Save every:        {save_every} episodes")
    print(f"Starting epsilon:  {resume_epsilon:.6f}")
    print(f"{'='*70}\n")

    # Initialize ROS and environment
    rclpy.init()
    env = GazeboEnv(linear_speed=speed, lock_step=True)
    agent = DDQNAgent(learning_rate=learning_rate)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(resolved_checkpoint, map_location=device)
    
    # Handle both new 'full' checkpoints (model+optimizer) and old 'legacy' checkpoints (model only)
    if isinstance(checkpoint_data, dict) and 'model_state' in checkpoint_data:
        print("Restoring Full Checkpoint (Model weights + Optimizer state)")
        agent.policy_net.load_state_dict(checkpoint_data['model_state'])
        agent.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
    else:
        print("Restoring Legacy Checkpoint (Model weights only)")
        agent.policy_net.load_state_dict(checkpoint_data)
        
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.target_net.eval()
    agent.epsilon = resume_epsilon

    # Restore Replay Buffer memory if available
    memory_file = resolved_checkpoint.with_name(resolved_checkpoint.stem + "_memory.pkl")
    if memory_file.exists():
        print(f"Restoring Replay Buffer from {memory_file.name}...")
        with open(memory_file, 'rb') as f:
            agent.memory = pickle.load(f)
        print(f"  -> Replay buffer restored with {len(agent.memory)} experiences.")
    else:
        print("! No Replay Buffer match found. Resuming with an empty memory.")

    # Restore training history
    reward_history = existing_rewards.tolist() if existing_rewards is not None else []
    action_history = existing_actions.tolist() if existing_actions is not None else []
    crash_history = existing_crashes.tolist() if existing_crashes is not None else []
    epsilon_history = existing_epsilons.tolist() if existing_epsilons is not None else []

    # Training loop
    try:
        for episode in range(current_episode + 1, args.target_episodes + 1):
            state, start_coords = env.reset()
            episode_reward = 0
            episode_actions = []

            for step in range(max_steps_per_episode):
                action_idx = agent.get_action(state)
                episode_actions.append(action_idx)

                next_state, reward, done, crash_coords = env.step(action_idx)
                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.train_step()

                state = next_state
                episode_reward += reward

                if done:
                    crash_history.append(crash_coords)
                    break

            reward_history.append(episode_reward)
            epsilon_history.append(agent.epsilon)

            if len(episode_actions) < max_steps_per_episode:
                episode_actions.extend([-1] * (max_steps_per_episode - len(episode_actions)))
            action_history.append(episode_actions)

            agent.update_epsilon()

            formatted_crash = f"({crash_coords[0]:.2f}, {crash_coords[1]:.2f})" if done else 'No'
            print(f"Episode {episode:04d}/{args.target_episodes} | Reward: {episode_reward:8.1f} | "
                  f"ε: {agent.epsilon:.3f} | Crash: {formatted_crash}")

            # Periodic checkpoint save
            if episode % save_every == 0 or episode == args.target_episodes:
                model_path = resolved_checkpoint.parent / f"ddqn_ep{episode:04d}.pth"
                checkpoint = {
                    'model_state': agent.policy_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict()
                }
                torch.save(checkpoint, model_path)
                
                # Save Replay Buffer
                memory_path = resolved_checkpoint.parent / f"ddqn_ep{episode:04d}_memory.pkl"
                with open(memory_path, 'wb') as f:
                    pickle.dump(agent.memory, f)

                start_idx = len(existing_rewards) if existing_rewards is not None else 0
                existing_rewards = append_and_save(rewards_path, existing_rewards, reward_history[start_idx:])

                start_idx = len(existing_actions) if existing_actions is not None else 0
                existing_actions = append_and_save(actions_path, existing_actions, action_history[start_idx:])

                start_idx = len(existing_crashes) if existing_crashes is not None else 0
                existing_crashes = append_and_save(crashes_path, existing_crashes, crash_history[start_idx:])

                start_idx = len(existing_epsilons) if existing_epsilons is not None else 0
                existing_epsilons = append_and_save(epsilons_path, existing_epsilons, epsilon_history[start_idx:])

                # Reload arrays to reflect appended data
                existing_rewards = np.load(rewards_path)
                existing_actions = np.load(actions_path, allow_pickle=True)
                existing_crashes = np.load(crashes_path, allow_pickle=True)
                existing_epsilons = np.load(epsilons_path)

                print(f"  Saved: {model_path.name}")

    except (KeyboardInterrupt, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "Unable to convert call argument" not in str(e):
            raise e
            
        print("\n--- Training restore interrupted by user (CTRL-C) ---")
        print("Saving full checkpoint and memory buffer before exiting...")
        
        interrupted_ep = episode if 'episode' in locals() else current_episode
        model_path = resolved_checkpoint.parent / f"ddqn_ep{interrupted_ep:04d}_interrupted.pth"
        
        checkpoint = {
            'model_state': agent.policy_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        
        memory_path = resolved_checkpoint.parent / f"ddqn_ep{interrupted_ep:04d}_interrupted_memory.pkl"
        with open(memory_path, 'wb') as f:
            pickle.dump(agent.memory, f)
            
        start_idx = len(existing_rewards) if existing_rewards is not None else 0
        append_and_save(rewards_path, existing_rewards, reward_history[start_idx:])
        start_idx = len(existing_actions) if existing_actions is not None else 0
        append_and_save(actions_path, existing_actions, action_history[start_idx:])
        start_idx = len(existing_crashes) if existing_crashes is not None else 0
        append_and_save(crashes_path, existing_crashes, crash_history[start_idx:])
        start_idx = len(existing_epsilons) if existing_epsilons is not None else 0
        append_and_save(epsilons_path, existing_epsilons, epsilon_history[start_idx:])
        
        print(f"Interrupted checkpoint safely stored at: {model_path.name}")

    finally:
        print("\nTraining operation finished.")
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
