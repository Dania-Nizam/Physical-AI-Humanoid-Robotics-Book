---
title: Spawning and Controlling Humanoid Robots in Gazebo
sidebar_position: 4
---

# Spawning and Controlling Humanoid Robots in Gazebo

## Overview

In this chapter, we'll focus on creating, spawning, and controlling humanoid robots in Gazebo. Humanoid robots present unique challenges in simulation due to their complex kinematics, multiple degrees of freedom, and the need for stable locomotion. We'll cover the entire process from designing the robot model to implementing control algorithms.

## Understanding Humanoid Robot Design

Humanoid robots have a human-like structure with:
- Torso and head
- Two arms with hands
- Two legs with feet
- Multiple joints for locomotion and manipulation

The key challenges in simulating humanoid robots include:
- Maintaining balance during locomotion
- Coordinating multiple joints for complex movements
- Managing center of mass for stability
- Implementing realistic gait patterns

## Creating a Humanoid Robot Model

Let's create a detailed humanoid robot model. Create `~/digital_twin_ws/src/digital_twin_examples/models/humanoid_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base Footprint -->
  <link name="base_footprint">
    <visual>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.0001"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Base Link -->
  <joint name="base_footprint_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="-0.15 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="15" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="15" velocity="2"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="-0.08 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.5"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="25" velocity="1.5"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="15" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="0.08 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.5"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="25" velocity="1.5"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="15" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joint State Publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=joint_states</remapping>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>neck_joint</joint_name>
      <joint_name>left_shoulder_joint</joint_name>
      <joint_name>left_elbow_joint</joint_name>
      <joint_name>right_shoulder_joint</joint_name>
      <joint_name>right_elbow_joint</joint_name>
      <joint_name>left_hip_joint</joint_name>
      <joint_name>left_knee_joint</joint_name>
      <joint_name>left_ankle_joint</joint_name>
      <joint_name>right_hip_joint</joint_name>
      <joint_name>right_knee_joint</joint_name>
      <joint_name>right_ankle_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Joint Position Controllers -->
  <gazebo>
    <plugin name="position_controller" filename="libgazebo_ros_joint_pose_publisher.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
      </ros>
    </plugin>
  </gazebo>
</robot>
```

## Spawning the Humanoid Robot

Now let's create a Python script to spawn the humanoid robot in Gazebo. Create `~/digital_twin_ws/src/digital_twin_examples/scripts/spawn_humanoid.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity
import sys
import os

class HumanoidSpawner(Node):
    def __init__(self):
        super().__init__('humanoid_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /spawn_entity not available, waiting again...')
        self.req = SpawnEntity.Request()

    def spawn_humanoid(self, robot_name, urdf_path, x=0.0, y=0.0, z=1.0):
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
    urdf_path = os.path.expanduser("~/digital_twin_ws/src/digital_twin_examples/models/humanoid_robot.urdf")

    spawner = HumanoidSpawner()
    spawner.spawn_humanoid('humanoid_robot', urdf_path, 0.0, 0.0, 1.0)

    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Controlling the Humanoid Robot

Let's create a controller for the humanoid robot. Create `~/digital_twin_ws/src/digital_twin_examples/scripts/humanoid_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.publisher = self.create_publisher(JointState, '/humanoid_robot/joint_states', 10)

        # Define all joint names for the humanoid robot
        self.joint_names = [
            'neck_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_step = 0.0

        # Timer to publish joint states at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions

        # Update time step
        self.time_step += 0.02

        # Simple walking gait pattern
        self.generate_walking_pattern()

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        self.publisher.publish(msg)

    def generate_walking_pattern(self):
        """Generate a simple walking pattern for the humanoid"""
        # Hip joints - create alternating movement for walking
        left_hip_idx = self.joint_names.index('left_hip_joint')
        right_hip_idx = self.joint_names.index('right_hip_joint')

        # Knee joints
        left_knee_idx = self.joint_names.index('left_knee_joint')
        right_knee_idx = self.joint_names.index('right_knee_joint')

        # Ankle joints
        left_ankle_idx = self.joint_names.index('left_ankle_joint')
        right_ankle_idx = self.joint_names.index('right_ankle_joint')

        # Walking pattern - alternating legs
        left_hip_phase = math.sin(self.time_step * 2.0)
        right_hip_phase = math.sin(self.time_step * 2.0 + math.pi)  # Opposite phase

        # Hip movement
        self.joint_positions[left_hip_idx] = left_hip_phase * 0.3
        self.joint_positions[right_hip_idx] = right_hip_phase * 0.3

        # Knee movement (bend knees when hip moves forward)
        self.joint_positions[left_knee_idx] = max(0, left_hip_phase * 0.5)
        self.joint_positions[right_knee_idx] = max(0, right_hip_phase * 0.5)

        # Ankle movement for balance
        self.joint_positions[left_ankle_idx] = -left_hip_phase * 0.1
        self.joint_positions[right_ankle_idx] = -right_hip_phase * 0.1

        # Arm movement to counterbalance
        left_shoulder_idx = self.joint_names.index('left_shoulder_joint')
        right_shoulder_idx = self.joint_names.index('right_shoulder_joint')
        left_elbow_idx = self.joint_names.index('left_elbow_joint')
        right_elbow_idx = self.joint_names.index('right_elbow_joint')

        self.joint_positions[left_shoulder_idx] = -right_hip_phase * 0.4  # Opposite to leg
        self.joint_positions[right_shoulder_idx] = -left_hip_phase * 0.4  # Opposite to leg
        self.joint_positions[left_elbow_idx] = abs(left_hip_phase) * 0.3
        self.joint_positions[right_elbow_idx] = abs(right_hip_phase) * 0.3

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Control: Balance and Stability

For more advanced control, we can implement a simple balance controller. Create `~/digital_twin_ws/src/digital_twin_examples/scripts/balance_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import numpy as np
import math

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Publishers and subscribers
        self.joint_pub = self.create_publisher(JointState, '/humanoid_robot/joint_states', 10)
        self.imu_sub = self.create_subscription(Imu, '/humanoid_robot/imu', self.imu_callback, 10)

        # Joint names
        self.joint_names = [
            'neck_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)
        self.target_positions = [0.0] * len(self.joint_names)

        # IMU data storage
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # PID controller parameters for balance
        self.kp = 2.0  # Proportional gain
        self.kd = 0.5  # Derivative gain

        # Timer for control loop
        self.timer = self.create_timer(0.01, self.balance_control_loop)  # 100 Hz

    def imu_callback(self, msg):
        """Callback to process IMU data"""
        # Convert quaternion to Euler angles (simplified)
        # In a real implementation, you'd use proper quaternion to Euler conversion
        self.roll = math.atan2(2.0 * (msg.orientation.w * msg.orientation.x + msg.orientation.y * msg.orientation.z),
                              1.0 - 2.0 * (msg.orientation.x * msg.orientation.x + msg.orientation.y * msg.orientation.y))
        self.pitch = math.asin(2.0 * (msg.orientation.w * msg.orientation.y - msg.orientation.z * msg.orientation.x))
        self.yaw = math.atan2(2.0 * (msg.orientation.w * msg.orientation.z + msg.orientation.x * msg.orientation.y),
                             1.0 - 2.0 * (msg.orientation.y * msg.orientation.y + msg.orientation.z * msg.orientation.z))

    def balance_control_loop(self):
        """Main balance control loop"""
        # Calculate balance corrections based on IMU data
        roll_correction = -self.kp * self.roll - self.kd * 0  # Simplified - no derivative in this example
        pitch_correction = -self.kp * self.pitch - self.kd * 0

        # Apply corrections to ankle joints for balance
        left_ankle_idx = self.joint_names.index('left_ankle_joint')
        right_ankle_idx = self.joint_names.index('right_ankle_joint')

        # Adjust ankle positions to counteract tilt
        self.target_positions[left_ankle_idx] = roll_correction
        self.target_positions[right_ankle_idx] = -roll_correction  # Opposite for balance

        # Also adjust hip joints slightly for balance
        left_hip_idx = self.joint_names.index('left_hip_joint')
        right_hip_idx = self.joint_names.index('right_hip_joint')

        self.target_positions[left_hip_idx] += pitch_correction * 0.1
        self.target_positions[right_hip_idx] += pitch_correction * 0.1

        # Publish joint states
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.target_positions

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launching the Humanoid Robot

Create a launch file to bring up the robot and controllers together. Create `~/digital_twin_ws/src/digital_twin_examples/launch/humanoid_robot.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    ld = LaunchDescription()

    # Arguments
    world = LaunchConfiguration('world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty',
        description='Choose one of the world files from `/usr/share/gazebo-11/worlds`'
    )

    # Launch Gazebo
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world],
        output='screen'
    )

    # Spawn the humanoid robot
    spawn_robot = Node(
        package='digital_twin_examples',
        executable='spawn_humanoid',
        name='spawn_humanoid',
        output='screen'
    )

    # Launch the controller
    controller = Node(
        package='digital_twin_examples',
        executable='humanoid_controller',
        name='humanoid_controller',
        output='screen'
    )

    # Add all actions to launch description
    ld.add_action(world_arg)
    ld.add_action(gazebo)
    ld.add_action(spawn_robot)
    ld.add_action(controller)

    return ld
```

## Testing the Humanoid Robot

To test the humanoid robot, follow these steps:

1. **Build the workspace:**
```bash
cd ~/digital_twin_ws
colcon build --packages-select digital_twin_examples
source install/setup.bash
```

2. **Launch Gazebo with a world:**
```bash
gz sim -r worlds/physics_lab.sdf
```

3. **Spawn the robot:**
```bash
python3 ~/digital_twin_ws/src/digital_twin_examples/scripts/spawn_humanoid.py
```

4. **Run the controller:**
```bash
python3 ~/digital_twin_ws/src/digital_twin_examples/scripts/humanoid_controller.py
```

## Troubleshooting Common Issues

### Robot Falls Over
- Check that all joint limits are appropriate
- Verify that inertial properties are realistic
- Ensure the center of mass is properly positioned

### Joints Don't Move Smoothly
- Check that the controller is publishing at an appropriate rate
- Verify that joint limits aren't too restrictive
- Ensure that effort values in URDF are appropriate

### Gazebo Performance Issues
- Reduce the number of joints being controlled simultaneously
- Lower the physics update rate in the world file
- Simplify collision meshes where possible

## Best Practices for Humanoid Control

### Stability First
- Always implement basic balance control before complex movements
- Use PID controllers for stable joint position control
- Monitor center of mass during movement

### Smooth Transitions
- Use trajectory interpolation for smooth joint movements
- Implement velocity and acceleration limits
- Avoid sudden changes in joint positions

### Simulation Accuracy
- Match real-world robot parameters as closely as possible
- Test controllers in simulation before applying to real robots
- Validate physics properties through known scenarios

## Summary

In this chapter, we've covered the complete process of creating, spawning, and controlling humanoid robots in Gazebo:
- Designing a detailed humanoid robot model with proper URDF
- Implementing spawning mechanisms using Gazebo services
- Creating controllers for basic and advanced movements
- Implementing balance control using IMU feedback
- Testing and troubleshooting common issues

The next chapter will introduce Unity for robotics visualization, where we'll explore how to create high-fidelity visualizations of our robots that complement the physics-accurate simulations in Gazebo.