# Quickstart Guide: NVIDIA Isaac Sim and Isaac ROS Integration

## Overview

This quickstart guide will help you set up and run your first Isaac Sim simulation with Isaac ROS integration. The guide assumes you have the required software installed (ROS 2 Kilted Kaiju, NVIDIA Isaac Sim 5.0, Isaac ROS 3.2) with an appropriate NVIDIA GPU.

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 30/40 series recommended)
- NVIDIA drivers (535 or newer)
- CUDA 12.x installed
- ROS 2 Kilted Kaiju installed and sourced
- Isaac Sim 5.0 installed
- Isaac ROS 3.2 packages installed
- Python 3.8+ with rclpy
- Git for version control

## Setup Steps

### 1. Verify System Requirements

First, verify your NVIDIA GPU and driver setup:

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify ROS 2 installation
source /opt/ros/kilted/setup.bash
ros2 topic list
```

### 2. Install Isaac Sim

If you haven't installed Isaac Sim yet:

```bash
# Isaac Sim is typically installed via Omniverse Launcher or direct download
# Verify installation
cd ~/.local/share/ov/pkg/isaac_sim-2023.1.1/  # or your installation path
python -c "import omni; print('Isaac Sim installation verified')"
```

### 3. Install Isaac ROS Packages

Install the Isaac ROS packages for your use case:

```bash
# Install Isaac ROS common packages
sudo apt update
sudo apt install ros-kilted-isaac-ros-common

# Install Isaac ROS perception packages
sudo apt install ros-kilted-isaac-ros-perception

# Install Isaac ROS navigation packages
sudo apt install ros-kilted-isaac-ros-navigation

# Install Isaac ROS SLAM packages
sudo apt install ros-kilted-isaac-ros-slam
```

### 4. Verify Isaac Sim ROS Bridge

Test the basic Isaac Sim-ROS bridge functionality:

```bash
# Launch Isaac Sim with ROS bridge
# This is typically done through the Isaac Sim application or launch files
```

### 5. Create Your First Isaac Sim Environment

#### 5.1. Create a Simple USD Scene

Create `simple_room.usd`:

```usd
#usda 1.0
(
    doc = "Simple room with a box"
    metersPerUnit = 1
)

def Xform "World"
{
    def Xform "Room"
    {
        def Cube "Floor"
        {
            extent = [(-5, -0.05, -5), (5, 0.05, 5)]
            size = 1
            xformOp:scale = (10, 0.1, 10)
        }

        def Cube "Box"
        {
            extent = [(-0.5, 0, -0.5), (0.5, 1, 0.5)]
            size = 1
            xformOp:translate = (0, 0.5, 2)
        }
    }
}
```

#### 5.2. Create a Simple Humanoid Robot USD

Create `simple_humanoid.usd`:

```usd
#usda 1.0
(
    doc = "Simple humanoid robot"
    metersPerUnit = 1
)

def Xform "HumanoidRobot"
{
    def Xform "Base"
    {
        def Capsule "Torso"
        {
            extent = [(-0.1, -0.3, -0.1), (0.1, 0.3, 0.1)]
            radius = 0.1
            height = 0.6
        }

        def Capsule "Head"
        {
            extent = [(-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)]
            radius = 0.1
            height = 0.2
            xformOp:translate = (0, 0.5, 0)
        }

        # Left Arm
        def Capsule "LeftUpperArm"
        {
            extent = [(-0.05, -0.2, -0.05), (0.05, 0.2, 0.05)]
            radius = 0.05
            height = 0.4
            xformOp:translate = (-0.3, 0.2, 0)
        }

        def Capsule "LeftLowerArm"
        {
            extent = [(-0.04, -0.15, -0.04), (0.04, 0.15, 0.04)]
            radius = 0.04
            height = 0.3
            xformOp:translate = (-0.3, -0.25, 0)
        }

        # Right Arm
        def Capsule "RightUpperArm"
        {
            extent = [(-0.05, -0.2, -0.05), (0.05, 0.2, 0.05)]
            radius = 0.05
            height = 0.4
            xformOp:translate = (0.3, 0.2, 0)
        }

        def Capsule "RightLowerArm"
        {
            extent = [(-0.04, -0.15, -0.04), (0.04, 0.15, 0.04)]
            radius = 0.04
            height = 0.3
            xformOp:translate = (0.3, -0.25, 0)
        }

        # Left Leg
        def Capsule "LeftThigh"
        {
            extent = [(-0.06, -0.3, -0.06), (0.06, 0.3, 0.06)]
            radius = 0.06
            height = 0.6
            xformOp:translate = (-0.15, -0.5, 0)
        }

        def Capsule "LeftShin"
        {
            extent = [(-0.05, -0.3, -0.05), (0.05, 0.3, 0.05)]
            radius = 0.05
            height = 0.6
            xformOp:translate = (-0.15, -0.9, 0)
        }

        # Right Leg
        def Capsule "RightThigh"
        {
            extent = [(-0.06, -0.3, -0.06), (0.06, 0.3, 0.06)]
            radius = 0.06
            height = 0.6
            xformOp:translate = (0.15, -0.5, 0)
        }

        def Capsule "RightShin"
        {
            extent = [(-0.05, -0.3, -0.05), (0.05, 0.3, 0.05)]
            radius = 0.05
            height = 0.6
            xformOp:translate = (0.15, -0.9, 0)
        }
    }
}
```

### 6. Launch Isaac Sim with ROS Integration

#### 6.1. Create a Launch Script

Create `launch_isaac_sim.py`:

```python
#!/usr/bin/env python3

import sys
import subprocess
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class IsaacSimLauncher(Node):
    def __init__(self):
        super().__init__('isaac_sim_launcher')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Timer to publish dummy joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_step = 0.0

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions

        # Simple oscillating motion
        self.time_step += 0.1
        for i, name in enumerate(self.joint_names):
            if 'hip' in name or 'knee' in name:
                self.joint_positions[i] = 0.2 * (i % 2) * 3.14159 * 0.2 * (1 + 0.3 * (i % 3) *
                    (0.5 + 0.5 * 3.14159 * self.time_step * 0.2))

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)

def main(args=None):
    print("Starting Isaac Sim launcher...")

    # Launch Isaac Sim (this is a simplified example)
    # In practice, you would use Isaac Sim's Python API or launch scripts
    print("Isaac Sim should be launched with ROS bridge enabled")
    print("See Isaac Sim documentation for detailed launch procedures")

    # Initialize ROS
    rclpy.init(args=args)
    launcher = IsaacSimLauncher()

    try:
        rclpy.spin(launcher)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        launcher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 7. Run Isaac ROS Perception Pipeline

#### 7.1. Create a Visual SLAM Launch File

Create `isaac_ros_slam_launch.py`:

```python
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    """Launch file for Isaac ROS Visual SLAM"""

    container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam',
                parameters=[{
                    'enable_rectified_pose': True,
                    'map_frame': 'map',
                    'odometry_frame': 'odom',
                    'base_frame': 'base_link',
                    'imu_frame': 'imu_link',
                    'camera_frame': 'camera_link',
                }],
                remappings=[
                    ('/visual_slam/imu', '/imu/data'),
                    ('/visual_slam/left/camera_info', '/camera/left/camera_info'),
                    ('/visual_slam/left/image', '/camera/left/image_rect_color'),
                    ('/visual_slam/right/camera_info', '/camera/right/camera_info'),
                    ('/visual_slam/right/image', '/camera/right/image_rect_color'),
                    ('/visual_slam/tracking/pose_graph', '/pose_graph'),
                    ('/visual_slam/visual_odometry', '/visual_odometry'),
                ],
            ),
        ],
        output='screen',
    )

    return launch.LaunchDescription([container])
```

### 8. Testing the Integration

1. Start ROS 2: `source /opt/ros/kilted/setup.bash`
2. Launch Isaac Sim with your scene
3. Start the ROS bridge in Isaac Sim
4. Run the perception pipeline: `ros2 launch path/to/isaac_ros_slam_launch.py`
5. Monitor topics: `ros2 topic echo /visual_odometry`

### 9. Troubleshooting Common Issues

#### Isaac Sim Not Launching
- Check NVIDIA GPU drivers are properly installed
- Verify CUDA installation and compatibility
- Ensure Isaac Sim license is valid

#### ROS Bridge Not Working
- Verify Isaac Sim ROS bridge extension is enabled
- Check that ROS 2 environment is sourced
- Ensure proper network configuration for bridge

#### Isaac ROS Packages Not Found
- Verify packages are installed via apt
- Check ROS 2 package path: `echo $AMENT_PREFIX_PATH`
- Ensure ROS 2 Kilted Kaiju is properly sourced

### 10. Next Steps

- Explore advanced Isaac Sim features and USD scene creation
- Implement custom perception pipelines with Isaac ROS
- Configure Nav2 for bipedal humanoid navigation
- Generate synthetic data for perception training
- Optimize performance for your specific use case

This quickstart provides the foundation for working with NVIDIA Isaac Sim and Isaac ROS. The following chapters will dive deeper into each aspect of the integration.