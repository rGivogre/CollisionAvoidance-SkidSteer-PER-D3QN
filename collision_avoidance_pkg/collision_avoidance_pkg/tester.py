#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import tty
import termios
from .gazebo_env import GazeboEnv

# Match your RL setup configuration
LINEAR_SPEED = 0.5  # Constant forward velocity (meters/second)

msg = """
======================================================
Skid-Steering Action Tester Node
======================================================
Press keys 1 to 10 to command discrete steering angles:
(Matches the angular velocity equations in your plots)

[1] : -0.80 rad/s (Hard Right)
[2] : -0.64 rad/s
[3] : -0.48 rad/s
[4] : -0.32 rad/s
[5] : -0.16 rad/s (Slight Right)
[6] :  0.00 rad/s (Straight Ahead)
[7] :  0.16 rad/s (Slight Left)
[8] :  0.32 rad/s
[9] :  0.48 rad/s
[10]:  0.64 rad/s (Hard Left - *Key '0' maps to 10*)

Press 'Spacebar' or 's' to Stop the robot.
Press 'r' or 'R' to trigger a Respawn command state.
Press 'CTRL+C' to exit safely.
======================================================
"""

class ActionTester(Node):

    def __init__(self):
        super().__init__('action_tester_node')
        # Change '/cmd_vel' to match your robot's specific topic if needed
        self.publisher_ = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info("Action Tester initialized. Awaiting keyboard inputs...")


    def getKey(self):
        """ Captures a single keystroke from the terminal non-blockingly. """
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        env = GazeboEnv()
        print(msg)
        try:
            while rclpy.ok():
                key = self.getKey()
                
                action_index = None
                
                # Check for keys 1 through 9
                if key in [str(i) for i in range(1, 10)]:
                    action_index = int(key) - 1 # Map to 0-indexed values (0 to 8)
                
                # Map keyboard button '0' to represent Action 10
                elif key == '0':
                    action_index = 9 # Map to 0-indexed value (9)
                
                # Spacebar or 's' commands an immediate stop
                elif key == ' ' or key == 's':
                    twist = Twist()
                    self.publisher_.publish(twist)
                    print(f"\r[STOP COMMANDED] Linear Vel: {twist.linear.x:.2f} m/s | Angular Vel: {twist.angular.z:.2f} rad/s" + " " * 20)
                    continue
                
                # NEW: Capture 'r' or 'R' to act as a placeholder for environmental respawns
                elif key.lower() == 'r':
                    
                    print("\n[RESPAWN TRIGGERED] Resetting motor velocity states...")
                    env.reset() # Reset for the next episode
                    print(f"BOOM! Collision.")
                    # Note: If your framework uses a Gazebo reset service, you can trigger it here.
                    continue
                
                # If CTRL+C is captured via terminal character mapping
                elif key == '\x03':
                    break

                # If a valid action index was triggered, compute target velocities
                if action_index is not None:
                    twist = Twist()
                    
                    # Keep a baseline forward movement so you can observe the turn radius
                    twist.linear.x = LINEAR_SPEED
                    
                    # Exact math formula from your plot labels: round(-0.8 + 0.16 * action_index, 2)
                    twist.angular.z = -0.8 + (0.16 * action_index)
                    
                    self.publisher_.publish(twist)
                    print(f"[ACTION KEY {action_index + 1:2}] Linear Velocity: {twist.linear.x:.2f} m/s | Angular Velocity: {twist.angular.z:.2f} rad/s")

        except Exception as e:
            self.get_logger().error(f"Error encountered in execution loop: {e}")
            
        finally:
            # Emergency Stop on Shutdown sequence
            twist = Twist()
            self.publisher_.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    rclpy.init(args=args)
    node = ActionTester()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()