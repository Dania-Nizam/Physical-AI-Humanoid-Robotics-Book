---
title: Setting Up Modern Gazebo with ROS 2 Integration
sidebar_position: 2
---

# Setting Up Modern Gazebo with ROS 2 Integration

## Overview

In this chapter, we'll set up Modern Gazebo (Jetty) with ROS 2 Kilted Kaiju integration. This foundation is essential for creating physics-accurate simulations of robotic systems. Modern Gazebo represents the latest evolution of the Gazebo simulation platform, offering improved performance, better ROS 2 integration, and enhanced physics capabilities.

## Prerequisites

Before starting this setup, ensure you have:
- A Linux system (Ubuntu 22.04 LTS recommended)
- ROS 2 Kilted Kaiju installed and properly configured
- Administrative access to install system packages
- Basic familiarity with terminal commands

## Installing ROS 2 Kilted Kaiju

If you haven't already installed ROS 2 Kilted Kaiju, follow these steps:

```bash
# Add the ROS 2 GPG key and repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package list and install ROS 2 Kilted Kaiju
sudo apt update
sudo apt install -y ros-kilted-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

Initialize rosdep:
```bash
sudo rosdep init
rosdep update
```

Source the ROS 2 environment:
```bash
source /opt/ros/kilted/setup.bash
```

To make this permanent, add it to your bashrc:
```bash
echo "source /opt/ros/kilted/setup.bash" >> ~/.bashrc
```

## Installing Modern Gazebo (Jetty)

Modern Gazebo (Jetty) is the latest version of the Gazebo simulation platform. Install it using:

```bash
# Install Gazebo packages
sudo apt install -y gz-jetty

# Install additional dependencies
sudo apt install -y gz-jetty-dev
```

Verify the installation:
```bash
gz --version
```

You should see output indicating Gazebo Garden or Jetty is installed.

## Installing ROS 2 Gazebo Integration Packages

To connect ROS 2 with Modern Gazebo, install the ros_gz packages:

```bash
sudo apt install -y ros-kilted-ros-gz
sudo apt install -y ros-kilted-ros-gz-bridge
sudo apt install -y ros-kilted-gz-sim
sudo apt install -y ros-kilted-gz-sim-ros2-control
```

## Setting Up the Workspace

Create a dedicated workspace for our digital twin projects:

```bash
mkdir -p ~/digital_twin_ws/src
cd ~/digital_twin_ws
source /opt/ros/kilted/setup.bash
colcon build
```

Add the workspace to your bashrc:
```bash
echo "source ~/digital_twin_ws/install/setup.bash" >> ~/.bashrc
```

## Testing the Installation

Let's verify that everything is working correctly by launching a simple Gazebo simulation:

```bash
# Source the ROS 2 environment
source /opt/ros/kilted/setup.bash

# Launch Gazebo with a simple world
gz sim
```

You should see the Gazebo simulation environment launch successfully. If you encounter any issues, check that:

1. ROS 2 is properly sourced
2. Gazebo packages are correctly installed
3. There are no permission issues

## Creating a Basic World File

Let's create a simple world file to test our setup:

Create `~/digital_twin_ws/src/digital_twin_examples/worlds/basic_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="basic_room">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 -0.4 -0.8</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
          <material>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.8 0.3 0.2 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

Launch the world:
```bash
gz sim ~/digital_twin_ws/src/digital_twin_examples/worlds/basic_room.sdf
```

## Setting Up ROS 2 Gazebo Bridge

The ROS 2 Gazebo bridge allows communication between ROS 2 nodes and Gazebo. Test the bridge with:

```bash
# Terminal 1: Launch Gazebo with a world that has ROS 2 bridge
source /opt/ros/kilted/setup.bash
gz sim -v 4 --render-engine ogre ~/digital_twin_ws/src/digital_twin_examples/worlds/basic_room.sdf

# Terminal 2: Test ROS 2 communication
source /opt/ros/kilted/setup.bash
ros2 topic list
```

You should see ROS 2 topics being published by Gazebo.

## Creating a Robot Model

Let's create a simple robot model to test our setup. Create `~/digital_twin_ws/src/digital_twin_examples/models/simple_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <link name="sensor_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="base_sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_link"/>
    <origin xyz="0 0 0.2"/>
  </joint>
</robot>
```

## Testing Robot Spawn

To test if we can spawn our robot in Gazebo, we'll create a simple Python script:

Create `~/digital_twin_ws/src/digital_twin_examples/scripts/spawn_robot.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity
import sys
import os

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /spawn_entity not available, waiting again...')
        self.req = SpawnEntity.Request()

    def spawn_robot(self, robot_name, urdf_path, x=0.0, y=0.0, z=1.0):
        # Read URDF file
        with open(urdf_path, 'r') as urdf_file:
            urdf_content = urdf_file.read()

        # Set up the request
        self.req.name = robot_name
        self.req.xml = urdf_content
        self.req.robot_namespace = ""

        # Set initial pose
        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z

        self.req.initial_pose = initial_pose

        # Send the request
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        if self.future.result() is not None:
            self.get_logger().info(f'Successfully spawned {robot_name}')
            return self.future.result()
        else:
            self.get_logger().error(f'Failed to spawn {robot_name}')
            return None

def main(args=None):
    rclpy.init(args=args)

    # Get the path to the URDF file
    urdf_path = os.path.expanduser("~/digital_twin_ws/src/digital_twin_examples/models/simple_robot.urdf")

    spawner = RobotSpawner()
    spawner.spawn_robot('simple_robot', urdf_path, 0.0, 0.0, 0.5)

    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Issues

### Gazebo Not Launching

If Gazebo fails to launch, check:
- Graphics drivers are properly installed
- X11 forwarding is enabled if using SSH
- GPU acceleration is available

### ROS 2 Bridge Not Working

If the ROS 2 bridge isn't functioning:
- Verify that ros_gz packages are installed
- Check that both ROS 2 and Gazebo environments are sourced
- Ensure the correct Gazebo plugins are loaded

### Robot Spawn Failures

If robot spawning fails:
- Verify URDF file syntax
- Check that Gazebo simulation is running
- Ensure the spawn service is available

## Summary

In this chapter, we've successfully set up Modern Gazebo with ROS 2 Kilted Kaiju integration. We've verified that:
- Gazebo launches correctly
- The ROS 2 bridge is functional
- We can create and test simple world and robot models

This foundation enables us to proceed with more complex simulation scenarios in the following chapters. The next chapter will focus on creating more sophisticated environments with detailed physics properties.