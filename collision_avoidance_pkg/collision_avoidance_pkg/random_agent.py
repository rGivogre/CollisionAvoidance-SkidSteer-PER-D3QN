import rclpy
import random
import time
from geometry_msgs.msg import Twist
from .gazebo_env import GazeboEnv # Import the environment

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize our environment
    env = GazeboEnv()
    
    print("Starting Random Agent test! Press CTRL+C to stop.")
    time.sleep(2) # Wait 2 seconds for Gazebo to stabilize
    
    done = False
    total_reward = 0
    
    try:
        while not done:
            # Choose a random action from 0 to 10
            action = random.randint(0, 10)
            
            # Execute the step in the environment
            state, reward, done = env.step(action)
            total_reward += reward
            
            print(f"Action: {action} | Reward: {reward} | State[0]: {state[0]:.2f}m")
            
            # If it crashed, stop the robot
            if done:
                print(f"BOOM! Collision detected. Total reward: {total_reward}")
                
                # Send a null action to stop the wheels
                stop_cmd = Twist()
                env.cmd_vel_pub.publish(stop_cmd)
                
    except KeyboardInterrupt:
        pass
    
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()