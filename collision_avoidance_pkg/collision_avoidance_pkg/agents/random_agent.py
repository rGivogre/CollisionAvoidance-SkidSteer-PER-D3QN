import rclpy
import random
import time
from geometry_msgs.msg import Twist
from ..gazebo_env import GazeboEnv   # Import the environment

def main(args=None):
    rclpy.init(args=args)

    # Initialize our environment
    env = GazeboEnv()
    
    print("Starting Random Agent test! Press CTRL+C to stop.")
    time.sleep(2) # Wait 2 seconds for Gazebo to stabilize

    try:
        epochs = 5 # Let's test 5 episodes
        for epoch in range(1, epochs + 1):
            state = env.reset() # Start of the episode
            done = False
            total_reward = 0
            
            print(f"--- Starting Epoch {epoch} ---")
            
            while not done:
                action = random.randint(0, 10)

                # Execute the step in the environment
                state, reward, done = env.step(action)
                total_reward += reward
                print(f"Action: {action} | Reward: {reward} | State[0]: {state[0]:.2f}m")
                
                if done:
                    env.reset() # Reset for the next episode
                    print(f"BOOM! Collision. Epoch {epoch} ended. Total reward: {total_reward}")
                    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()