import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
import numpy as np
import random
import math
import time
import json
import os

# Environment Constants, revise them as needed for our specific Gazebo world and robot configuration
ROBOT_NAME = 'skid_bot'             # Name of the entity in Gazebo
MAX_LIDAR_RANGE = 10          
NUM_LIDAR_RAYS = 50                 # State size
FRONT_COLLISION_THRESHOLD = 0.95    # Derived from max sweep radius (0.922m) + tolerance
SIDE_COLLISION_THRESHOLD = 0.45     # Robot half-width (0.4m) + 0.05m tolerance
ANGULAR_SPEED_BASE = -0.8       
ANGULAR_SPEED_STEP = 0.16       

class GazeboEnv(Node):
    def __init__(self, linear_speed=0.3, lock_step=False, map_name='map2'):
        super().__init__('gazebo_env_node', allow_undeclared_parameters=True, parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        # use_sim_time is crucial for synchronizing with Gazebo's clock, especially when running in fast mode during training. It ensures that all time-based operations (like waiting for sensor updates) are aligned with the simulation time rather than real-world time.

        # Publisher on /demo/cmd_vel to move the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        # Subscriber on /scan to read the LiDAR scan
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        # Subscriber for Odometry to track position
        self.odom_sub = self.create_subscription(Odometry, '/demo/odom', self.odom_callback, qos_profile_sensor_data)

        # Service client to teleport the robot (SetEntityState)
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
            
        # Service clients for synchronizing physics (Lock-step execution)
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')
        while not self.pause_client.wait_for_service(timeout_sec=1.0) or \
              not self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /pause_physics and /unpause_physics services...')

        # Internal state variables
        self.state = np.zeros(NUM_LIDAR_RAYS)
        self.collision = False

        self.map_name = map_name
        self.safe_spawn_zones = []
        self._load_spawn_zones()
        
        # Initialize coordinate tracking variables
        self.start_coords = (0.0, 0.0)
        self.current_x = 0.0
        self.current_y = 0.0
        
        self.linear_speed = linear_speed
        self.lock_step = lock_step

        self.new_scan_received = False
    
    def _load_spawn_zones(self):
        """Loads spawn zones from config/spawn_zones.json based on the map_name."""
        # Find path to the workspace relative config relative to current working directory
        config_path = os.path.join(os.getcwd(), 'config', 'spawn_zones.json')
        
        try:
            with open(config_path, 'r') as f:
                all_zones = json.load(f)
                
            if self.map_name in all_zones:
                self.safe_spawn_zones = all_zones[self.map_name]
                self.get_logger().info(f"Loaded {len(self.safe_spawn_zones)} spawn zones for map '{self.map_name}'.")
            else:
                self.get_logger().warn(f"Map '{self.map_name}' not found in {config_path}. Falling back to default (0,0).")
                self.safe_spawn_zones = [[0.0, 0.0, 0.0, 0.0]]
        except FileNotFoundError:
            self.get_logger().error(f"Config file not found at {config_path}. Falling back to default (0,0).")
            self.safe_spawn_zones = [[0.0, 0.0, 0.0, 0.0]]
        except Exception as e:
            self.get_logger().error(f"Failed to load spawn zones: {e}. Falling back to default (0,0).")
            self.safe_spawn_zones = [[0.0, 0.0, 0.0, 0.0]]

    def scan_callback(self, msg):
        """Processes LiDAR data every time it arrives and evaluates collision status."""
        # Convert measurements to a numpy array
        ranges = np.array(msg.ranges)

        # Extract uniformly distributed measurements
        # np.linspace selects NUM_LIDAR_RAYS evenly spaced indices from the array's total length
        indices = np.linspace(0, len(ranges) - 1, NUM_LIDAR_RAYS, dtype=int)
        raw_state = ranges[indices]

        # The laser sometimes returns "infinite" if it sees nothing. Limit to max range.
        raw_state[np.isinf(raw_state)] = MAX_LIDAR_RANGE
        raw_state[np.isnan(raw_state)] = MAX_LIDAR_RANGE
    
        
        # LiDAR span: 270 degrees (-135 to +135). Resolution ~5.4 deg/ray.
        right_rays = raw_state[:19]     # -135 to -32 degrees
        front_rays = raw_state[19:31]   # -32 to +32 degrees
        left_rays = raw_state[31:]      # +32 to +135 degrees

        # Evaluate Inevitable Collision States
        if (np.min(front_rays) < FRONT_COLLISION_THRESHOLD) or (np.min(left_rays) < SIDE_COLLISION_THRESHOLD) or (np.min(right_rays) < SIDE_COLLISION_THRESHOLD):
            self.collision = True
        else:
            self.collision = False
            
        # Normalize to [0, 1] by dividing by max range, for better neural network performance
        self.state = raw_state / MAX_LIDAR_RANGE
        self.new_scan_received = True   # Flag to indicate that a new scan has been processed, used for synchronization in step() and reset()

    def odom_callback(self, msg):
        """Callback to constantly update current global position."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def _drain_lidar_queue(self):
        """Actively flush any old LiDAR messages still in the ROS 2 queue. Uses timeout_sec=0.0 to process messages instantly and return."""
        while True:
            self.new_scan_received = False
            rclpy.spin_once(self, timeout_sec=0.0)
            if not self.new_scan_received:
                break

    def _wait_for_new_scan(self, required_scans=1):
        """Blocks execution until 'required_scans' new LiDAR scans are received, with a Watchdog."""
        for _ in range(required_scans):
            self.new_scan_received = False
            start_time = time.time()

            while rclpy.ok() and not self.new_scan_received:
                rclpy.spin_once(self, timeout_sec=0.001)

                if time.time() - start_time > 2.0: # Watchdog timeout of 2 seconds, which is generous for a 10Hz LiDAR. If we hit this, it likely means the physics engine is paused and we need to unpause it to get new scans.
                    self.get_logger().warn("Watchdog trigger: LiDAR timeout! Forcing physics unpause...")
                    self._unpause_physics()
                    start_time = time.time()  # Reset watchdog timer after unpausing

    def _pause_physics(self):
        """Freezes the Gazebo physics engine."""
        self.pause_client.call_async(Empty.Request())

    def _unpause_physics(self):
        """Unfreezes the Gazebo physics engine."""
        self.unpause_client.call_async(Empty.Request())

    def _compute_reward(self, normalized_state, action):

        front_rays = normalized_state[19:31]
        
        # Calculate key metrics
        front_clearance_mean = float(np.mean(front_rays))
        min_clearance = float(np.min(normalized_state))
        omega = ANGULAR_SPEED_BASE + (ANGULAR_SPEED_STEP * action)

        r_base = 0.1    # Base Survival (Small incentive to keep the episode running)
        
        # Rewards the robot proportionally to how clear the path ahead is.
        r_forward = 3.0 * front_clearance_mean  # Max value: 3.0 (if completely clear, remember the state is normalized to [0,1])
        
        # Penalizes rotation ONLY when the path ahead is clear. Skid-steer kinematics degrade performance when turning unnecessarily.
        r_steer = 1.5 * abs(omega) * front_clearance_mean   # If front is blocked (front_clearance_mean almost 0), turning is free.
        
        # Applies a linear penalty only when closer than a critical threshold.
        normalized_crash_threshold = FRONT_COLLISION_THRESHOLD / MAX_LIDAR_RANGE
        safe_margin = normalized_crash_threshold + 0.1      # safety boundary starts 1 meter (0.1 normalized) before the actual crash point
        if min_clearance < safe_margin:
            r_danger = 2.0 * ((safe_margin - min_clearance) / (safe_margin - normalized_crash_threshold))  # Linearly increases from 0 (at safe_margin) to 2.0 (at crash threshold)
            r_danger = min(r_danger, 2.0)  # Cap the danger penalty at 2.0 (handles side collisions that can be much closer than the front threshold)
        else:
            r_danger = 0.0
            
        reward = r_base + r_forward - r_steer - r_danger
        return float(reward)


    def step(self, action):
        """Receives the action (0-10), moves the robot and calculates the reward."""

        if self.lock_step:
            self._unpause_physics() # Let the physics engine run to execute our action
        
        # Move the robot using parameterized speeds (paper's formula)
        vel_cmd = Twist()
        vel_cmd.linear.x = self.linear_speed
        vel_cmd.angular.z = ANGULAR_SPEED_BASE + (ANGULAR_SPEED_STEP * action)
        
        self.cmd_vel_pub.publish(vel_cmd)
        self._drain_lidar_queue()   # QUEUE DRAIN: Flush old frames generated before the command had a physical effect
        
        self._wait_for_new_scan(required_scans=2)   # Wait for two new scans to ensure the robot has moved enough to reflect the action's consequences in the LiDAR data.
    
        # Freeze the gazebo universe exactly when we get our observation
        if self.lock_step:
            self._pause_physics()   # This gives PyTorch unlimited real-world time to compute gradients safely.

        # Calculate the reward
        if self.collision:
            reward = -20.0  # Mitigated collision penalty, to prevent gradient explosions in Huber Loss
            done = True
            crash_coords = (self.current_x, self.current_y)
        else:
            reward = self._compute_reward(self.state, action)
            done = False
            crash_coords = None
            
        return self.state.copy(), reward, done, crash_coords
    
    def reset(self):
        """Resets the robot teleporting it to a random safe location and returns the initial state."""

        if self.lock_step:
            self._unpause_physics()     # to allow the teleport drop and stabilization
        
        # Stop the robot's movement
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        MAX_SPAWN_ATTEMPTS = 10
        SAFE_SPAWN_THRESHOLD = FRONT_COLLISION_THRESHOLD + 0.15

        for attempt in range(MAX_SPAWN_ATTEMPTS):
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
            req.state.pose.position.z = 0.16    # Spawn the base_link above the ground, given that robot's wheels have a radius of 0.15. 
            req.state.pose.orientation.x = 0.0
            req.state.pose.orientation.y = 0.0
            req.state.pose.orientation.z = math.sin(yaw / 2.0)
            req.state.pose.orientation.w = math.cos(yaw / 2.0)
            
            # Call the service to teleport the robot
            future = self.set_state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            self._drain_lidar_queue()   # Actively flush any lingering messages from the previous crash

            self._wait_for_new_scan(required_scans=5)   # Ensure the robot is fully settled in the new position by waiting for 5 scans, approximately 0.5 seconds at 10Hz. This helps to avoid any residual effects from the teleportation and ensures the state is stable before starting the episode.
            
            raw_min_distance = np.min(self.state ) * MAX_LIDAR_RANGE
            if raw_min_distance >= SAFE_SPAWN_THRESHOLD:
                break  # Found a safe spawn point, exit the loop
            else:
                self.get_logger().warn(f"Spawn attempt {attempt + 1}: Unsafe spawn point detected (min distance: {raw_min_distance:.2f}m). Retrying...")
        else:
            self.get_logger().error("Failed to find a safe spawn point after multiple attempts. Proceeding with the last attempted spawn location, but this may lead to immediate collisions.")
        
        self.collision = False
        start_coords = (self.current_x, self.current_y)
        
        if self.lock_step:
            self._pause_physics()   

        return self.state.copy(), start_coords