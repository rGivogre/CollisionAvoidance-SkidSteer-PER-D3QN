import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    """
    Modified launch file to spawn the Skidbot (4-wheel skid steer) 
    instead of the TurtleBot3.
    """

    # 1. Retrieve the absolute paths to the required ROS 2 packages
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_collision_avoidance = get_package_share_directory('collision_avoidance_pkg')
    
    # Change: Point to your new skid_bot package
    pkg_skid_bot = get_package_share_directory('skid_bot')

    # 2. Define the path to your custom world
    world_file = os.path.join(pkg_collision_avoidance, 'worlds', 'map2_resized.world')

    # 3. Command to start 'gzserver'
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # 4. Command to start 'gzclient'
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # 5. Define the path to the Skidbot model
    # Most community models use URDF/Xacro. 
    # The skid_bot repo typically stores it in 'urdf/skid_bot.urdf'
    urdf_file = os.path.join(pkg_skid_bot, 'description', 'robot.urdf.xacro')
    
    # Note: If the file is just a .urdf and not .xacro, use that path.
    # If it is a .xacro, Gazebo might need it processed, but spawn_entity 
    # can often handle the processed XML from the robot_state_publisher.
    # For now, we point to the description file.

    # 6. Node to spawn the skid_bot entity
    spawn_skidbot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'skid_bot',     # Unique name for the entity
            '-topic', 'robot_description', # Spawns the robot from the robot_description topic
            '-x', '-2.0',
            '-y', '0.5',
            '-z', '0.05'               # Slightly higher to ensure wheels touch ground correctly
        ],
        output='screen',
    )

    # 7. Robot State Publisher (Required for Skidbot to load the URDF properly)
    # This node converts Xacro/URDF into a format Gazebo and RViz understand
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_skid_bot, 'launch', 'rsp.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Return the compiled list
    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher, # Adds the robot description to the ROS ecosystem
        spawn_skidbot_cmd      # Spawns the robot using that description
    ])