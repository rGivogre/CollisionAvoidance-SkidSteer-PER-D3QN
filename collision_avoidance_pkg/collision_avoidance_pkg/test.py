import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os
from .gazebo_env import GazeboEnv
from .agents.DDQN_agent import QNetwork, STATE_SIZE, ACTION_SIZE

# Note: We create a test class that inherits from Node to natively handle ROS 2 parameters
class SkidbotTestManager(Node):
    def __init__(self):
        super().__init__('skidbot_test_manager')
        
        # Declare ROS 2 parameters with their default values
        self.declare_parameter('model_path', 'models/ddqn_skidbot_ep3000.pth')
        self.declare_parameter('num_test_episodes', 5)
        self.declare_parameter('max_steps_per_episode', 500)
        
        # Note: The 'test_world' parameter is usually passed to the Gazebo launch file, 
        # but we declare it here in case your logic or logs need to track it.
        self.declare_parameter('test_world', 'map2.world')

        # Retrieve the actual values (defaults or passed from command line)
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.num_episodes = self.get_parameter('num_test_episodes').get_parameter_value().integer_value
        self.max_steps = self.get_parameter('max_steps_per_episode').get_parameter_value().integer_value
        self.test_world = self.get_parameter('test_world').get_parameter_value().string_value

def main():
    rclpy.init()
    
    # Initialize the test parameter manager and the Gazebo environment
    test_manager = SkidbotTestManager()
    env = GazeboEnv()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TEST] Using device: {device}")
    print(f"[TEST] Configured world: {test_manager.test_world}")
    print(f"[TEST] Total episodes: {test_manager.num_episodes}")
    print(f"[TEST] Max steps per episode: {test_manager.max_steps}")
    
    # Configure the neural network architecture
    policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
    
    # Load weights from the dynamic path retrieved from parameters
    if os.path.exists(test_manager.model_path):
        print(f"[TEST] Loading model from: {test_manager.model_path}")
        policy_net.load_state_dict(torch.load(test_manager.model_path, map_location=device))
        policy_net.eval()
    else:
        print(f"[TEST] Error: The model file '{test_manager.model_path}' does not exist.")
        env.destroy_node()
        test_manager.destroy_node()
        rclpy.shutdown()
        return

    print("\n--- Starting Skidbot Test Phase ---")

    for episode in range(1, test_manager.num_episodes + 1):
        state, start_coords = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nStarting Test Episode {episode}...")

        for step in range(test_manager.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            next_state, reward, done, crash_coords = env.step(action_idx)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                print(f"Episode terminated by collision after {steps} steps.")
                break
        
        print(f"Episode {episode} Result -> Steps completed: {steps}, Total Reward: {episode_reward}")

    print("\nTest phase completed.")
    env.destroy_node()
    test_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()