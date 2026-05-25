import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
import numpy as np
import time
import random
import math

class GazeboEnv(Node):
    def __init__(self):
        super().__init__('gazebo_env_node')
        
        # Publisher on /cmd_vel to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber on /scan to read the LiDAR scan
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Service client to teleport the robot (SetEntityState)
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # Internal state variables
        self.state = np.zeros(50) # Initialize the 50-element array
        self.collision = False
        self.min_distance = 0.20  # If an obstacle is closer than 20 cm, it's a collision

        # TODO: Ask if is better to do normalization here or in the agent.

        # TODO: Change this approach to one with rectangular safe zones instead of points. So that is more general for training.
        # (X, Y, Yaw in radiants)
        self.safe_spawn_points = [
            (-2.0, 0.5, 0.0),        
            (1.0, 1.0, 1.57),        
            (-1.5, -1.5, 3.14),      
            (1.5, -0.5, -1.57),
            (0.0, -2.0, 0.78),
            (2.0, 0.0, -0.78)
        ]
    
    def scan_callback(self, msg):
        """Processes LiDAR data every time it arrives."""
        # Convert measurements to a numpy array
        ranges = np.array(msg.ranges)
        
        # The laser sometimes returns "infinite" if it sees nothing. Limit to 5 meters.
        ranges[np.isinf(ranges)] = 5.0
        ranges[np.isnan(ranges)] = 5.0
        
        # Extract exactly 50 uniformly distributed measurements
        # np.linspace selects 50 evenly spaced indices from the array's total length
        indices = np.linspace(0, len(ranges) - 1, 50, dtype=int)
        self.state = ranges[indices]
        
        # Check if a collision occurred (if the shortest ray is below the threshold)
        if np.min(self.state) < self.min_distance:
            self.collision = True
        else:
            self.collision = False

    def step(self, action):
        """Receives the action (0-10), moves the robot and calculates the reward."""
        
        # Move the robot using the formula from the paper
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15  # Fixed forward speed (e.g. 0.15 m/s)
        vel_cmd.angular.z = -0.8 + (0.16 * action)
        
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
            
        return self.state.copy(), reward, done
    
    def reset(self):
        """Resets the robot teleporting it to a random safe location and returns the initial state."""
        # Stop the robot's movement
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rclpy.spin_once(self, timeout_sec=0.1)

        #TODO: Change this approach to one with rectangular safe zones instead of points.
        # Choose a random spawn point from the predefined safe locations
        x, y, yaw = random.choice(self.safe_spawn_points)
        
        # Create the request to teleport the robot
        req = SetEntityState.Request()
        req.state.name = 'turtlebot3_burger'
        req.state.reference_frame = 'world'
        
        # Set the robot's position and orientation
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.01    # Slightly above the ground to avoid spawning issues
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Call the service to teleport the robot
        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.1)
        
        self.collision = False
        return self.state.copy()