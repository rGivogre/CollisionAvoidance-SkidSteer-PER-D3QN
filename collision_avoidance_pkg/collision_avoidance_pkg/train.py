import rclpy
import numpy as np
import torch
import os
from .gazebo_env import GazeboEnv
from .agent import DDQNAgent

# Training parameters (from the paper)
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 500
SAVE_EVERY = 100  # Save the model every x episodes

def main():
    # Initialize ROS 2 system, environment node, and the DDQN agent
    rclpy.init()
    env = GazeboEnv()
    agent = DDQNAgent()
    
    # Setup project directories for saving models and plot data
    current_dir = os.path.dirname(os.path.abspath(__file__))    # Find the root of the project (CollisionAvoidance-RL) regardless of cwd
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    models_dir = os.path.join(project_root, 'models')
    plot_data_dir = os.path.join(project_root, 'plot_data')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    
    reward_history = []
    action_history = []
    epsilon_history = []
    print("--- Starting Skidbot Training (DDQN) ---")

    for episode in range(1, MAX_EPISODES + 1):
        # Reset the environment (Initialize state s1)
        state = env.reset()
        episode_reward = 0
        episode_actions = []  # Track actions taken this episode

        for step in range(MAX_STEPS_PER_EPISODE):
            # Select an action using epsilon-greedy policy
            action_idx = agent.get_action(state)
            episode_actions.append(action_idx)  # Log action
            
            # Execute action in Gazebo and observe next state and reward
            next_state, reward, done = env.step(action_idx)
            
            # Store transition in memory
            agent.store_transition(state, action_idx, reward, next_state, done)
            
            # Train the neural network (sample minibatch & update)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
            if done:
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
        print(f"Episode: {episode}/{MAX_EPISODES}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

        # Save the model and log data
        if episode % SAVE_EVERY == 0:
            model_path = os.path.join(models_dir, f"ddqn_skidbot_ep{episode}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            
            # Save data arrays for plot.py
            np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
            np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history))
            np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
            
            print(f"Model and logs saved at episode {episode}")

    # Final cleanup
    print("Training completed.")
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()