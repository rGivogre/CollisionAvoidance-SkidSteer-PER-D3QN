import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class GazeboEnv(Node):
    def __init__(self):
        super().__init__('gazebo_env_node')
        
        # Publisher on /cmd_vel to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber on /scan to read the LiDAR scan
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Internal state variables
        self.state = np.zeros(50) # Initialize the 50-element array
        self.collision = False
        self.min_distance = 0.20  # If an obstacle is closer than 20 cm, it's a collision

    def scan_callback(self, msg):
        """This function processes LiDAR data every time it arrives."""
        # Convert measurements to a numpy array
        ranges = np.array(msg.ranges)
        
        # The laser sometimes returns "infinite" if it sees nothing. Limit to 3.5 meters.
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 3.5
        
        # The paper extracts exactly 50 uniformly distributed measurements
        # np.linspace selects 50 evenly spaced indices from the array's total length
        indices = np.linspace(0, len(ranges) - 1, 50, dtype=int)
        self.state = ranges[indices]
        
        # Check if a collision occurred (if the shortest ray is below the threshold)
        if np.min(self.state) < self.min_distance:
            self.collision = True
        else:
            self.collision = False

    def step(self, action):
        """This function receives the action (0-10), moves the robot and calculates the reward."""
        
        # Move the robot using the formula from the paper
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15  # Fixed forward speed (e.g. 0.15 m/s)
        vel_cmd.angular.z = -0.8 + (0.16 * action) # The action ranges from 0 to 10
        
        self.cmd_vel_pub.publish(vel_cmd)
        
        # Advance time in ROS 2 for 0.1 seconds to allow sensor updates
        # This is the "trick" to wait for the robot to move and the laser to read the new state
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Calculate the reward
        if self.collision:
            reward = -1000
            done = True
        else:
            reward = 5
            done = False
            
        return self.state, reward, done