import json
import rclpy
import numpy as np
import torch
import os
import argparse
import pickle
from datetime import datetime
from .gazebo_env import GazeboEnv
from .agents.D3QN_agent import DuelingDDQNAgent

# Training parameters (from the paper)
MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 1100
SAVE_EVERY = 250  # Save the model every x episodes

def main():
    parser = argparse.ArgumentParser(description="Skidbot D3QN Training Node")
    parser.add_argument('--speed', type=float, default=0.3, help='Linear velocity (LINEAR_SPEED) for the robot navigation')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate for the neural network')

    args, _ = parser.parse_known_args() # Extract known args so it doesn't crash on ROS 2 internal arguments
    speed = args.speed
    lr = args.learning_rate

    # Initialize ROS 2 system, environment node, and the D3QN agent
    rclpy.init()
    env = GazeboEnv(linear_speed=speed, lock_step=True)
    agent = DuelingDDQNAgent()
    
    # Setup project directories for saving models and plot data (assuming execution from repo root)
    models_base_dir = 'models'
    plot_data_base_dir = 'plot_data'
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = run_timestamp
    
    # Create subfolders for the specific run using the timestamp
    models_dir = os.path.join(models_base_dir, run_name)
    plot_data_dir = os.path.join(plot_data_base_dir, run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    run_config = {
        'run_name': run_name,
        'start_time': run_timestamp,
        'agent_key': 'D3QN',
        'speed': speed,
        'learning_rate': lr,
        'max_steps_per_episode': MAX_STEPS_PER_EPISODE,
        'save_every': SAVE_EVERY,
        'models_dir': models_dir,
        'plot_data_dir': plot_data_dir,
        'checkpoint_template': f"d3qn_ep{{episode:04d}}.pth",
    }

    for config_dir in [models_dir, plot_data_dir]:
        with open(os.path.join(config_dir, 'runconfig.json'), 'w') as config_file:
            json.dump(run_config, config_file, indent=2)

    reward_history = []
    action_history = []
    crash_history = []
    start_history = []
    epsilon_history = []
    print("--- Starting Skidbot Training (D3QN - Dueling DDQN) ---")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            # Reset the environment (Initialize state s1)
            state, start_coords = env.reset()
            start_history.append(start_coords)
            episode_reward = 0
            episode_actions = []  # Track actions taken this episode

            for step in range(MAX_STEPS_PER_EPISODE):
                # Select an action using epsilon-greedy policy
                action_idx = agent.get_action(state)
                episode_actions.append(action_idx)  # Log action
                
                # Execute action in Gazebo and observe next state and reward
                next_state, reward, done, crash_coords = env.step(action_idx)
                
                # Store transition in memory
                agent.store_transition(state, action_idx, reward, next_state, done)
                
                # Train the neural network (sample minibatch & update)
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    crash_history.append(crash_coords)  # Log crash coordinates
                    break
            
            # Log episode reward and epsilon for plotting
            reward_history.append(episode_reward)
            epsilon_history.append(agent.epsilon)
            
            # Pad episode actions if the robot crashed early so it fits cleanly into a matrix later
            if len(episode_actions) < MAX_STEPS_PER_EPISODE:
                episode_actions.extend([-1] * (MAX_STEPS_PER_EPISODE - len(episode_actions)))
            action_history.append(episode_actions)

            # Update epsilon (exploration) after every episode
            agent.update_epsilon()

            # Show progress 
            formatted_crash = f"({crash_coords[0]:.2f}, {crash_coords[1]:.2f})" if done else 'No'
            print(f"Episode: {episode}/{MAX_EPISODES}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Crash: {formatted_crash}")

            # Save the model and log data
            if (episode % SAVE_EVERY  == 0):
                model_path = os.path.join(models_dir, f"d3qn_ep{episode:04d}.pth")
                checkpoint = {
                    'model_state': agent.policy_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict()
                }
                torch.save(checkpoint, model_path)
                
                # Save Replay Buffer
                memory_path = os.path.join(models_dir, f"d3qn_ep{episode:04d}_memory.pkl")
                with open(memory_path, 'wb') as f:
                    pickle.dump(agent.memory, f)
                
                # Save data arrays for plot.py with standard names inside the timestamp folder
                np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
                np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history))
                np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
                np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
                
                print(f"Model and logs saved at episode {episode}")
                print(f"  -> Model: {os.path.abspath(model_path)}")
                print(f"  -> Run Folder: {os.path.abspath(plot_data_dir)}")

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user (CTRL-C) ---")
        print("Saving full checkpoint and memory buffer before exiting...")
        
        # Determine current episode
        interrupted_ep = episode if 'episode' in locals() else 0
        
        # Save model and optimizer
        model_path = os.path.join(models_dir, f"d3qn_ep{interrupted_ep:04d}_interrupted.pth")
        checkpoint = {
            'model_state': agent.policy_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        
        # Save Replay Buffer
        memory_path = os.path.join(models_dir, f"d3qn_ep{interrupted_ep:04d}_interrupted_memory.pkl")
        with open(memory_path, 'wb') as f:
            pickle.dump(agent.memory, f)
            
        # Save final history logs
        np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
        np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history))
        np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
        np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
        
        print(f"Interrupted checkpoint safely stored at: {model_path}")

    finally:
        # Final cleanup
        print("Cleaning up and terminating nodes safely...")
        env.destroy_node()
        rclpy.shutdown()
        print("Training session closed.")

if __name__ == '__main__':
    main()
