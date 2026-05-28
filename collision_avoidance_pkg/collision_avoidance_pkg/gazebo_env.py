import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
import numpy as np
import random
import math

# Environment Constants, revise them as needed for our specific Gazebo world and robot configuration
ROBOT_NAME = 'skid_bot'         # Name of the entity in Gazebo
MAX_LIDAR_RANGE = 10          
NUM_LIDAR_RAYS = 50             # State size
COLLISION_DISTANCE = 0.40       # If an obstacle is closer than this, it's considered a collision (in meters)
ANGULAR_SPEED_BASE = -0.8       
ANGULAR_SPEED_STEP = 0.16       

class GazeboEnv(Node):
    def __init__(self, linear_speed=0.3):
        super().__init__('gazebo_env_node', allow_undeclared_parameters=True, parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        # use_sim_time is crucial for synchronizing with Gazebo's clock, especially when running in fast mode during training. It ensures that all time-based operations (like waiting for sensor updates) are aligned with the simulation time rather than real-world time.

        # Publisher on /demo/cmd_vel to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        # Subscriber on /scan to read the LiDAR scan
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        # Service client to teleport the robot (SetEntityState)
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
            
        # Add Subscriber for Odometry to track position
        self.odom_sub = self.create_subscription(Odometry, '/demo/odom', self.odom_callback, qos_profile_sensor_data)

        # Internal state variables
        self.state = np.zeros(NUM_LIDAR_RAYS)
        self.collision = False

        # Rectangular safe zones for spawn: (x_min, x_max, y_min, y_max)
        self.safe_spawn_zones = [
            (1.806280, 9.096520, 2.245540, 4.153355),    
            (-21.616300, -19.892700, 3.597740, 10.221505)
        ]
        
        # Initialize coordinate tracking variables
        self.start_coords = (0.0, 0.0)
        self.current_x = 0.0
        self.current_y = 0.0
        
        self.linear_speed = linear_speed

        self.new_scan_received = False
    
    def scan_callback(self, msg):
        """Processes LiDAR data every time it arrives."""
        # Convert measurements to a numpy array
        ranges = np.array(msg.ranges)
        
        # The laser sometimes returns "infinite" if it sees nothing. Limit to max range.
        ranges[np.isinf(ranges)] = MAX_LIDAR_RANGE
        ranges[np.isnan(ranges)] = MAX_LIDAR_RANGE
        
        # Extract uniformly distributed measurements
        # np.linspace selects NUM_LIDAR_RAYS evenly spaced indices from the array's total length
        indices = np.linspace(0, len(ranges) - 1, NUM_LIDAR_RAYS, dtype=int)
        raw_state = ranges[indices]
        
        # Check if a collision occurred using raw distance (before normalization)
        if np.min(raw_state) < COLLISION_DISTANCE:
            self.collision = True
        else:
            self.collision = False
            
        # Normalize to [0, 1] by dividing by max range, for better neural network performance
        self.state = raw_state / MAX_LIDAR_RANGE
        self.new_scan_received = True

    def odom_callback(self, msg):
        """Callback to constantly update current global position."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def step(self, action):
        """Receives the action (0-10), moves the robot and calculates the reward."""
        
        # Move the robot using parameterized speeds (paper's formula)
        vel_cmd = Twist()
        vel_cmd.linear.x = self.linear_speed
        vel_cmd.angular.z = ANGULAR_SPEED_BASE + (ANGULAR_SPEED_STEP * action)
        
        self.cmd_vel_pub.publish(vel_cmd)
        
        # Advance time in ROS 2 until a new sensor reading is received (Event-Driven)
        self.new_scan_received = False
        while rclpy.ok() and not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Calculate the reward
        if self.collision:
            reward = -1000
            done = True
            crash_coords = (self.current_x, self.current_y)
        else:
            reward = 5
            done = False
            crash_coords = None
            
        return self.state.copy(), reward, done, crash_coords
    
    def reset(self):
        """Resets the robot teleporting it to a random safe location and returns the initial state."""
        # Stop the robot's movement
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        self.new_scan_received = False
        while rclpy.ok() and not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.01)

        # Choose a random spawn zone from the predefined safe areas
        x_min, x_max, y_min, y_max = random.choice(self.safe_spawn_zones)
        
        # Make the spawn continuous across the area and randomize the orientation
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        yaw = random.uniform(-math.pi, math.pi)
        
        # Create the request to teleport the robot
        req = SetEntityState.Request()
        req.state.name = ROBOT_NAME
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

        # QUEUE DRAIN (Best Practice): 
        # Actively flush any old LiDAR messages still in the queue after a crash
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.01)

        # Wait for a fresh scan at the new location
        self.new_scan_received = False
        while rclpy.ok() and not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        self.collision = False
        start_coords = (self.current_x, self.current_y)
        
        return self.state.copy(), start_coords