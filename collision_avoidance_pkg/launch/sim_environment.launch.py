import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    """
    Generates the launch description for the simulation environment.
    It tells ROS 2 exactly which nodes and background processes to start. Can choose between different worlds and whether to launch the Gazebo GUI.
    """
    # CONFIGURABILE VARIABLES (DEFAULTS) 
    default_world_name = 'multi_maps.world'
    default_gui_flag = 'false'

    # Create LaunchConfigurations (allows overriding via terminal arguments)
    world_file_arg = LaunchConfiguration('world_file')
    gui_arg = LaunchConfiguration('gui')

    # Retrieve the absolute paths to the packages
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_collision_avoidance = get_package_share_directory('collision_avoidance_pkg')
    pkg_skidbot = get_package_share_directory('skid_bot')

    # Evaluate the exact path to the selected .world file dynamically
    world_path = PathJoinSubstitution([pkg_collision_avoidance, 'worlds', world_file_arg])

    # Command to start 'gzserver' (Core Physics - Always runs)
    # Note: We pass the world file as an argument to gzserver, so it loads the correct environment.
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # Command to start 'gzclient' (GUI - Only runs if gui_arg is 'true')
    # this allows us to visualize the simulation when needed.
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(gui_arg)
    )

    # Robot State Publisher (Required to convert Xacro to URDF and publish transforms)
    # This evaluates the skid_bot xacro file dynamically and provides it to the /robot_description topic
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_skidbot, 'launch', 'rsp.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Node to spawn the robot entity inside the simulation
    # We spawn from the topic published by robot_state_publisher, not from a static file
    spawn_robot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'skid_bot',          # Unique name for the entity in Gazebo
            '-topic', 'robot_description', 
            '-x', '2.1',  
            '-y', '3.0',  
            '-z', '0.01'    
        ],
        output='screen',
    )

    # Return the compiled list of tasks
    return LaunchDescription([
        # Declarations expose these arguments to the ros2 launch CLI
        DeclareLaunchArgument('world_file', default_value=default_world_name, description='World filename to load'),
        DeclareLaunchArgument('gui', default_value=default_gui_flag, description='Launch Gazebo UI? true/false'),
        
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher,  # adds the robot description to the ROS ecosystem
        spawn_robot_cmd         # spawns the robot in Gazebo using that description
    ])