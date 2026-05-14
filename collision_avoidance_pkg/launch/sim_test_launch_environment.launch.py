import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    """
    This function generates the launch description for the simulation environment.
    It tells ROS 2 exactly which nodes and background processes to start.
    """

    # Retrieve the absolute paths to the required ROS 2 packages
    # This prevents hardcoding paths, making the code work on any computer
    # Paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_collision_avoidance = get_package_share_directory('collision_avoidance_pkg')
    pkg_husky_description = get_package_share_directory('husky_description')

    xacro_file = os.path.join(pkg_husky_description, 'urdf', 'husky.urdf.xacro')
    # This renders the Xacro into a XML string
    robot_description_xml = xacro.process_file(xacro_file).toxml()

    # Define the exact path to our custom training world (the .world file)
    world_file = os.path.join(pkg_collision_avoidance, 'worlds', 'map2.world')

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
    # urdf_file = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')
    
# 4. Robot State Publisher (Required for Husky)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_xml, 'use_sim_time': True}]
    )

    # 5. Spawn the Entity
    # We use '-topic' because the state publisher is now "streaming" the robot model
    spawn_husky_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'husky',
            '-topic', 'robot_description',
            '-x', '-2.0',
            '-y', '0.5',
            '-z', '0.2' # Spawning slightly higher because Husky is taller
        ],
        output='screen',
    )

    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        node_robot_state_publisher,
        spawn_husky_cmd
    ])