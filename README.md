# Collision Avoidance RL

This ROS 2 package provides a Deep Reinforcement Learning environment for a custom Skid-Steer robot collision avoidance using Gazebo.

## Prerequisites

### 1. Python & PyTorch
Install `pip` and the PyTorch CPU version:

```bash
sudo apt install python3-pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Skid-Steer Robot Package
This environment relies on a custom Skid-Steer robot model. You need to clone its repository into your workspace's `src` folder (right next to the `CollisionAvoidance-RL` folder):

```bash
cd ~/ros_project_ws/src
git clone https://github.com/odobot/ROS2-SKID-STEER-DRIVE-ROBOT.git
```

### 3. Adding the LiDAR Sensor
The original skid-steer repository does not come with a LiDAR sensor configured. We have provided an updated setup file in our repository (`xacro/skid.xacro`). After cloning both repositories, you must overwrite the original robot description file with ours:

```bash
cp ~/ros_project_ws/src/CollisionAvoidance-RL/xacro/skid.xacro ~/ros_project_ws/src/ROS2-SKID-STEER-DRIVE-ROBOT/src/skid_bot/description/skid.xacro
```

*(Note: After completing these steps, proceed to build the workspace and open a new terminal to source it).*

## Building the Workspace

To build the package, navigate to the root of your ROS 2 workspace (e.g., `~/ros_project_ws`). 
For active development, always use the `--symlink-install` flag. This flag creates symbolic links to your Python scripts, launch files, and worlds instead of mechanically copying them. As a result, you won't need to rebuild the workspace every time you edit an existing file—your changes will take effect immediately upon saving!

If you previously built without this flag or encounter caching issues, clean the workspace first:

```bash
cd ~/ros_project_ws
rm -rf build/ install/ log/
colcon build --symlink-install
```

*(Note: You must still re-run `colcon build --symlink-install` if you create brand new files or modify `setup.py` / `package.xml`).*

## How to Run

To run the simulation and the agent, you need to open **two different terminals**.

### Terminal 1 (Start the Simulation):
From this terminal, you launch Gazebo. You can customize the simulation using **launch arguments**.

```bash
ros2 launch collision_avoidance_pkg sim_environment.launch.py
```

**Launch Arguments Available:**
- `gui` *(default: `false`)*: Set this to `true` to open the Gazebo graphical interface. Keeping it `false` (headless mode) is optimal for high-speed RL training because it doesn't waste GPU resources rendering graphics.
- `world_file` *(default: `multi_maps.world`)*: The name of the world file to load from the package's `worlds/` folder.

*Example with custom arguments:*
```bash
ros2 launch collision_avoidance_pkg sim_environment.launch.py gui:=true world_file:=map2.world
```

### Terminal 2 (Run the Agent):
Wait for Gazebo to fully load the physics server in the first terminal, then run:
```bash
ros2 run collision_avoidance_pkg random_agent
```