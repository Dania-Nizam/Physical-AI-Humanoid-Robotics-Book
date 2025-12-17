# Quickstart Guide: Gazebo & Unity Digital Twin

## Overview

This quickstart guide will help you set up and run your first digital twin simulation using Gazebo for physics simulation and Unity for high-fidelity visualization. The guide assumes you have the required software installed (ROS 2 Kilted Kaiju, Modern Gazebo Jetty, Unity 2022.3+, Unity Robotics Hub).

## Prerequisites

- ROS 2 Kilted Kaiju installed and sourced
- Modern Gazebo (Jetty) installed
- Unity Hub with Unity 2022.3+ LTS
- Unity Robotics Hub package installed
- Python 3.8+ with rclpy
- Git for version control

## Setup Steps

### 1. Clone and Initialize the Repository

```bash
git clone https://github.com/your-org/ai-book-digital-twin.git
cd ai-book-digital-twin
source /opt/ros/kilted_kaiju/setup.bash
```

### 2. Install ROS 2 Gazebo Integration

```bash
sudo apt update
sudo apt install ros-kilted-kaiju-ros-gz ros-kilted-kaiju-ros-gz-bridge
```

### 3. Verify Gazebo Installation

```bash
gz sim
```

You should see the Gazebo simulation environment launch successfully.

### 4. Create Your First Digital Twin Environment

#### 4.1. Create a Simple World File

Create `worlds/simple_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="simple_room">
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

    <model name="table">
      <pose>0 0.5 0.4 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

#### 4.2. Create a Simple Robot Model

Create `models/simple_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
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

  <!-- Torso -->
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
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="base_torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.35"/>
  </joint>

  <!-- Head -->
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

  <joint name="torso_head_joint" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_thigh">
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
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_left_thigh_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="-0.1 0 -0.3"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
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
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_thigh_shin_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Right Leg -->
  <link name="right_thigh">
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
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_right_thigh_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="0.1 0 -0.3"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
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
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_thigh_shin_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_left_upper_arm_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="-0.2 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_upper_arm_lower_arm_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.15"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="torso_right_upper_arm_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.2 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_upper_arm_lower_arm_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.15"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <!-- LiDAR Sensor -->
  <link name="lidar_sensor">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="torso_lidar_joint" type="fixed">
    <parent link="torso"/>
    <child link="lidar_sensor"/>
    <origin xyz="0 0 0.25"/>
  </joint>

  <!-- IMU Sensor -->
  <link name="imu_sensor">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="torso_imu_joint" type="fixed">
    <parent link="torso"/>
    <child link="imu_sensor"/>
    <origin xyz="0 0 0.1"/>
  </joint>
</robot>
```

### 5. Launch the Simulation

#### 5.1. Start the Gazebo Simulation

```bash
# Source ROS 2
source /opt/ros/kilted_kaiju/setup.bash

# Launch Gazebo with your world
gz sim worlds/simple_room.sdf
```

#### 5.2. Spawn the Robot in Gazebo

Create a Python script to spawn the robot:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import os
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import time

# For Gazebo spawn service
from gazebo_msgs.srv import SpawnEntity

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
    urdf_path = os.path.join(os.getcwd(), 'models/simple_humanoid.urdf')

    spawner = RobotSpawner()
    spawner.spawn_robot('simple_humanoid_robot', urdf_path, 0.0, 0.0, 1.0)

    spawner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 5.3. Control the Robot

Create a simple controller script:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)

        # Timer to publish joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.joint_names = [
            'torso_left_thigh_joint', 'left_thigh_shin_joint',
            'torso_right_thigh_joint', 'right_thigh_shin_joint',
            'torso_left_upper_arm_joint', 'left_upper_arm_lower_arm_joint',
            'torso_right_upper_arm_joint', 'right_upper_arm_lower_arm_joint'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_step = 0.0

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions

        # Simple oscillating motion
        self.time_step += 0.1
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = 0.5 * (i % 2) * 3.14159 * 0.5 * (1 + 0.5 * (i % 3) *
                (0.5 + 0.5 * 3.14159 * self.time_step * 0.5))

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 6. Unity Integration

#### 6.1. Set Up Unity Project

1. Open Unity Hub and create a new 3D project
2. Import the Unity Robotics Hub package via Package Manager
3. Import the URDF Importer package
4. Import your robot model (URDF) into Unity

#### 6.2. Create ROS Connection

1. Create an empty GameObject and add the "ROS Connection" component
2. Configure the connection settings (typically localhost:10000)
3. Create a script to subscribe to joint states and update the robot model

Example Unity C# script for ROS connection:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class JointStateSubscriber : MonoBehaviour
{
    [SerializeField]
    private Dictionary<string, Transform> jointNameToTransform = new Dictionary<string, Transform>();

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<sensor_msgs_JointState>("/joint_states", JointStateCallback);

        // Populate the joint dictionary (assign in inspector or through code)
        // Example: jointNameToTransform.Add("torso_left_thigh_joint", leftThighTransform);
    }

    void JointStateCallback(sensor_msgs_JointState jointState)
    {
        for (int i = 0; i < jointState.name.Count; i++)
        {
            if (jointNameToTransform.ContainsKey(jointState.name[i]))
            {
                Transform jointTransform = jointNameToTransform[jointState.name[i]];

                // Apply the joint position to the transform
                // This is a simplified example - you may need to adapt based on your joint type
                jointTransform.localEulerAngles = new Vector3(
                    jointState.position[i] * Mathf.Rad2Deg,
                    jointTransform.localEulerAngles.y,
                    jointTransform.localEulerAngles.z
                );
            }
        }
    }
}
```

### 7. Running the Complete Digital Twin

1. Start ROS 2: `source /opt/ros/kilted_kaiju/setup.bash`
2. Launch Gazebo: `gz sim worlds/simple_room.sdf`
3. Spawn the robot using the Python script
4. Start the controller to move the robot
5. In Unity, run the scene with the ROS connection and joint state subscriber
6. The Unity visualization should now mirror the physics simulation in Gazebo

## API Usage Example

Once you have the simulation running, you can use the API to control it:

```bash
# Create a new environment
curl -X POST http://localhost:8080/environments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hospital_corridor",
    "description": "Hospital corridor simulation",
    "gazebo_world_file": "worlds/hospital_corridor.sdf",
    "unity_scene_file": "Scenes/HospitalCorridor.unity"
  }'

# Get the environment
curl -X GET http://localhost:8080/environments/{environment_id}

# Spawn a robot
curl -X POST http://localhost:8080/environments/{environment_id}/robots \
  -H "Content-Type: application/json" \
  -d '{
    "name": "atlas_robot",
    "urdf_file": "models/atlas.urdf"
  }'

# Start simulation
curl -X POST http://localhost:8080/environments/{environment_id}/simulate
```

## Troubleshooting

### Common Issues:

1. **Gazebo won't start**: Ensure ROS 2 is sourced and Gazebo packages are installed
2. **Robot not spawning**: Check that URDF file is valid and paths are correct
3. **Unity-ROS connection fails**: Verify that ROS TCP Connector is running and ports match
4. **Joint states not updating**: Check that controller is publishing to correct topic

### Verification Steps:

1. Check ROS 2 nodes: `ros2 node list`
2. Check ROS 2 topics: `ros2 topic list`
3. Verify Gazebo models: `gz model -m`
4. Test ROS connection: `ros2 topic echo /joint_states`

## Next Steps

- Explore advanced sensor simulation (LiDAR, depth cameras, IMUs)
- Implement more complex robot models
- Add Unity visualization enhancements
- Create custom Gazebo plugins for specific behaviors
- Develop more sophisticated control algorithms