import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    """
    This function generates the launch description for the simulation environment.
    It tells ROS 2 exactly which nodes and background processes to start.
    """

    # Retrieve the absolute paths to the required ROS 2 packages
    # This prevents hardcoding paths, making the code work on any computer
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_collision_avoidance = get_package_share_directory('collision_avoidance_pkg')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # Define the exact path to your custom training world (the .world file)
    # Uncomment the line below and replace 'map2.world' with the name of your world file when we will have a custom one
    # world_file = os.path.join(pkg_collision_avoidance, 'worlds', 'map2.world')

    world_file = os.path.join(pkg_turtlebot3_gazebo, 'worlds', 'turtlebot3_world.world') # IGNORE - This is the default world, we will replace it with our custom one

    # Command to start 'gzserver': the core physics engine of Gazebo
    # We pass our custom 'world_file' as an argument so it loads our maze
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # Command to start 'gzclient': the Graphical User Interface (GUI) of Gazebo
    # This allows us to visually see the robot and the environment (optional but helpful for debugging)
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # Define the path to the 3D model (SDF file) of the TurtleBot3 Burger
    urdf_file = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')
    
    # Node to spawn the robot entity inside the running Gazebo simulation
    spawn_turtlebot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3_burger',
            '-file', urdf_file,
            '-x', '-2.0',  # Initial X coordinate in the world
            '-y', '0.5',  # Initial Y coordinate in the world
            '-z', '0.01'  # Spawn slightly above ground to prevent physics glitches
        ],
        output='screen',
    )

    # Return the compiled list of actions to execute simultaneously
    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        spawn_turtlebot_cmd
    ])