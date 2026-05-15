# Collision Avoidance RL

This ROS 2 package provides a Deep Reinforcement Learning environment for TurtleBot3 collision avoidance using Gazebo.

## Prerequisites

Install `pip` and the PyTorch CPU version:

```bash
sudo apt install python3-pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Building the Workspace

Always remember to build the package from the root of your ROS 2 workspace (e.g., `ros_project_ws`) every time you pull the repo, add file or modify them.

```bash
cd ~/ros_project_ws
colcon build --symlink-install
```
### Pro Tip for Development: `--symlink-install`

During development, it is highly recommended to use the `--symlink-install` flag. This creates a shortcut (symlink) to your Python scripts instead of copying them. That way, any changes you make to your Python code (or `.launch.py` and `.world` files) are applied immediately upon saving, without needing to run `colcon build` again!

To switch to this method, you should first clean your workspace by deleting the old static build files, and then build with the flag:

```bash
cd ~/ros_project_ws
rm -rf build/ install/ log/
colcon build --symlink-install
```

*(Note: You will still need to run `colcon build --symlink-install` if you create completely new files or modify the `setup.py` file).*

## How to Run

To run the simulation and the agent, you need to open **two different terminals**.

**Terminal 1 (Start the Simulation):**
```bash
ros2 launch collision_avoidance_pkg sim_environment.launch.py
```

**Terminal 2 (Run the Agent):**
Wait for Gazebo to fully load in the first terminal, then run:
```bash
ros2 run collision_avoidance_pkg random_agent
```