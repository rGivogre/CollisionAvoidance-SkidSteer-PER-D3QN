import os
import tempfile
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def setup_gazebo_world(context, *args, **kwargs):
    """
    OpaqueFunction to evaluate the launch arguments and dynamically 
    modify the physics update rate inside the loaded world file.
    """
    world_filename = LaunchConfiguration('world_file').perform(context)
    speedup = LaunchConfiguration('speedup').perform(context).lower()

    pkg_collision_avoidance = get_package_share_directory('collision_avoidance_pkg')
    original_world_path = os.path.join(pkg_collision_avoidance, 'worlds', world_filename)

    # Read the original map
    with open(original_world_path, 'r') as f:
        world_content = f.read()

    # Apply the speedup logic (0 = unlimited speed, 1000 = RTF 1.0)
    update_rate = '0' if speedup == 'true' else '1000'
    
    # Replace the <real_time_update_rate> tag natively in the XML string
    world_content = re.sub(
        r'<real_time_update_rate>.*?</real_time_update_rate>', 
        f'<real_time_update_rate>{update_rate}</real_time_update_rate>', 
        world_content
    )

    # Save the modified world to a temporary folder
    tmp_world_path = os.path.join(tempfile.gettempdir(), f"custom_speed_{update_rate}_{world_filename}")
    with open(tmp_world_path, 'w') as f:
        f.write(world_content)

    # Command to start 'gzserver' with the modified world file. This will launch the Gazebo server with our custom physics settings.
    # Return the IncludeLaunchDescription for gzserver pointing to the TEMPORARY world
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': tmp_world_path}.items()
    )
    
    return [gzserver_cmd]

def generate_launch_description():
    """
    Generates the launch description for the simulation environment.
    It tells ROS 2 exactly which nodes and background processes to start. Can choose between different worlds and whether to launch the Gazebo GUI.
    """
    # CONFIGURABILE VARIABLES (DEFAULTS) 
    default_world_name = 'multi_maps.world'
    default_gui_flag = 'false'
    default_speedup = 'true'

    # Create LaunchConfigurations (allows overriding via terminal arguments)
    gui_arg = LaunchConfiguration('gui')

    # Retrieve the absolute paths to the packages
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_skidbot = get_package_share_directory('skid_bot')

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
            '-z', '0.01',
            '-timeout', '100'    
        ],
        output='screen',
    )

    # Return the compiled list of tasks
    return LaunchDescription([
        # Declarations expose these arguments to the ros2 launch CLI
        DeclareLaunchArgument('world_file', default_value=default_world_name, description='World filename to load'),
        DeclareLaunchArgument('gui', default_value=default_gui_flag, description='Launch Gazebo UI? true/false'),
        DeclareLaunchArgument('speedup', default_value=default_speedup, description='Run physics UNLIMITED (true) or REAL-TIME (false)'),
        
        OpaqueFunction(function=setup_gazebo_world),
        gzclient_cmd,
        robot_state_publisher,  # adds the robot description to the ROS ecosystem
        spawn_robot_cmd         # spawns the robot in Gazebo using that description
    ])