import rclpy
import numpy as np
import torch
import os
from datetime import datetime
from .gazebo_env import GazeboEnv
from .agents.DDQN_agent import DDQNAgent

# Training parameters (from the paper)
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 5000
SAVE_EVERY = 1000  # Save the model every x episodes

def main():
    # Initialize ROS 2 system, environment node, and the DDQN agent
    rclpy.init()
    env = GazeboEnv()
    agent = DDQNAgent()
    
    # Setup project directories for saving models and plot data (assuming execution from repo root)
    models_dir = 'models'
    plot_data_dir = 'plot_data'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    reward_history = []
    action_history = []
    crash_history = []
    start_history = []
    epsilon_history = []
    print("--- Starting Skidbot Training (DDQN) ---")

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
        print(f"Episode: {episode}/{MAX_EPISODES}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}, Crash: {formatted_crash}")

        # Save the model and log data
        if (episode % SAVE_EVERY  == 0) or (episode == 10):
            model_path = os.path.join(models_dir, f"ddqn_ep{episode}_{run_timestamp}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            
            # Save data arrays for plot.py
            np.save(os.path.join(plot_data_dir, f'rewards_{run_timestamp}.npy'), np.array(reward_history))
            np.save(os.path.join(plot_data_dir, f'actions_{run_timestamp}.npy'), np.array(action_history))
            np.save(os.path.join(plot_data_dir, f'crashes_{run_timestamp}.npy'), np.array(crash_history))
            np.save(os.path.join(plot_data_dir, f'epsilons_{run_timestamp}.npy'), np.array(epsilon_history))
            
            print(f"Model and logs saved at episode {episode}")
            print(f"  -> Model: {os.path.abspath(model_path)}")
            print(f"  -> Data folder: {os.path.abspath(plot_data_dir)}")

    # Final cleanup
    print("Training completed.")
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()