---
title: Launch Files and Composing Systems
description: Organizing and launching multiple ROS 2 nodes with launch files
tags: [ros2, launch, composition, system]
wordCount: 1250
---

# Launch Files and Composing Systems

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the purpose and benefits of launch files in ROS 2
- Create Python launch files to manage multiple nodes
- Configure nodes with parameters, remappings, and custom configurations
- Use launch conditions and arguments for flexible system composition
- Implement node composition for better performance
- Debug and troubleshoot launch file issues

## Introduction to Launch Files

Launch files in ROS 2 provide a way to start multiple nodes with a single command, along with their configurations, parameters, and connections. They are essential for:

- **System composition**: Starting complete robotic systems with a single command
- **Configuration management**: Setting parameters and configurations for multiple nodes
- **Development workflow**: Standardizing how systems are launched during development
- **Deployment**: Ensuring consistent node configurations across different environments

## Basic Launch File Structure

Launch files in ROS 2 are Python scripts that use the `launch` library to define and configure nodes. Here's a basic example:

```python
# minimal_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker',
            output='screen'
        ),
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='listener',
            output='screen'
        )
    ])
```

## Launch Arguments

Launch arguments allow you to customize launch files at runtime:

```python
# launch_with_args.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Use launch configuration in node
    return LaunchDescription([
        use_sim_time,
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        )
    ])
```

## Node Configuration Options

Launch files support extensive node configuration:

```python
# configured_nodes_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='robot1',
        description='Name of the robot'
    )

    return LaunchDescription([
        robot_name,
        Node(
            package='my_robot_package',
            executable='navigation_node',
            name=['nav_', LaunchConfiguration('robot_name')],
            namespace=LaunchConfiguration('robot_name'),
            parameters=[
                {'use_sim_time': False},
                {'planner_frequency': 1.0},
                {'controller_frequency': 20.0}
            ],
            remappings=[
                ('/cmd_vel', 'cmd_vel'),
                ('/scan', 'laser_scan'),
                ('/map', 'map')
            ],
            arguments=['--ros-args', '--log-level', 'info'],
            output='screen'
        )
    ])
```

## Conditional Launch

You can use conditions to control which nodes are launched:

```python
# conditional_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    use_camera = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Launch camera node'
    )

    use_lidar = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Launch lidar node'
    )

    return LaunchDescription([
        use_camera,
        use_lidar,
        Node(
            package='image_tools',
            executable='cam2image',
            name='camera_node',
            condition=IfCondition(LaunchConfiguration('use_camera')),
            output='screen'
        ),
        Node(
            package='velodyne_driver',
            executable='velodyne_node',
            name='lidar_node',
            condition=IfCondition(LaunchConfiguration('use_lidar')),
            output='screen'
        )
    ])
```

## Launch Substitutions

Launch substitutions provide dynamic values in launch files:

```python
# substitutions_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_file = DeclareLaunchArgument(
        'config_file',
        default_value=[PathJoinSubstitution([FindPackageShare('my_package'), 'config', 'default.yaml'])],
        description='Path to config file'
    )

    return LaunchDescription([
        config_file,
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[LaunchConfiguration('config_file')],
            output='screen'
        )
    ])
```

## Composable Nodes (Node Composition)

Node composition allows multiple nodes to run in the same process, reducing communication overhead:

```python
# composition_launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name=' perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='image_tools',
                plugin='image_tools::Cam2Image',
                name='cam2image',
                parameters=[{'width': 320, 'height': 240}]
            ),
            ComposableNode(
                package='image_tools',
                plugin='image_tools::ShowImage',
                name='showimage',
                parameters=[{'width': 320, 'height': 240}]
            )
        ],
        output='screen'
    )

    return LaunchDescription([container])
```

## Launch File Organization

For complex systems, organize launch files in a package structure:

```
my_robot_package/
├── launch/
│   ├── robot.launch.py          # Main robot launch
│   ├── navigation.launch.py     # Navigation subsystem
│   └── perception.launch.py     # Perception subsystem
├── config/
│   ├── navigation.yaml          # Navigation parameters
│   ├── sensors.yaml             # Sensor parameters
│   └── controllers.yaml         # Controller parameters
└── nodes/
    ├── navigation_node.py
    └── controller_node.py
```

## Launch File Best Practices

1. **Modular design**: Break complex systems into smaller, reusable launch files
2. **Use arguments**: Make launch files flexible with launch arguments
3. **Parameter files**: Use YAML files for complex parameter configurations
4. **Documentation**: Comment launch files to explain the system architecture
5. **Default values**: Provide sensible defaults for all arguments
6. **Error handling**: Use appropriate output settings for debugging

## Common Launch Commands

```bash
# Launch a file
ros2 launch my_package my_launch_file.py

# Launch with arguments
ros2 launch my_package my_launch_file.py use_sim_time:=true robot_name:=robot1

# List active launch processes
ros2 launch --list-processes

# Launch with specific log level
ros2 launch my_package my_launch_file.py --ros-args --log-level info
```

## Debugging Launch Files

Common debugging techniques:

1. **Check output**: Use `output='screen'` to see node output
2. **Log levels**: Adjust logging levels for detailed information
3. **Validate syntax**: Check Python syntax and import statements
4. **Check dependencies**: Ensure all required packages are installed
5. **Use launch introspection**: Use `ros2 launch --list-processes` to see what's running

## Launch File Patterns

### Include Other Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('navigation_package'),
            '/launch/navigation.launch.py'
        ])
    )

    return LaunchDescription([
        navigation_launch
    ])
```

### Event Handling

```python
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    # Launch node that triggers another when it exits
    controller_node = Node(
        package='my_package',
        executable='controller',
        name='controller'
    )

    cleanup_node = Node(
        package='my_package',
        executable='cleanup',
        name='cleanup'
    )

    # Register event handler
    event_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_node,
            on_exit=[cleanup_node]
        )
    )

    return LaunchDescription([
        controller_node,
        event_handler
    ])
```

## Summary

Launch files are essential for composing complex ROS 2 systems. They provide a clean, organized way to start multiple nodes with their configurations, parameters, and connections. Understanding launch files is crucial for developing, testing, and deploying ROS 2 applications effectively.

In the next chapter, we'll explore URDF (Unified Robot Description Format) fundamentals for modeling robot kinematics.