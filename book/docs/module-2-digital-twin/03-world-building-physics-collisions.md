---
title: World Building - Physics, Gravity, and Collisions in Gazebo
sidebar_position: 3
---

# World Building: Physics, Gravity, and Collisions in Gazebo

## Overview

In this chapter, we'll explore the fundamentals of creating simulation environments in Gazebo with a focus on physics properties, gravity settings, and collision detection. A well-designed simulation world is crucial for accurate robot testing and validation, as it determines how objects interact with each other and with the robot.

## Understanding Gazebo World Structure

Gazebo worlds are defined using SDF (Simulation Description Format) files, which specify:
- Physics properties (gravity, friction, damping)
- Lighting conditions
- Static and dynamic objects
- Terrain and environment models
- Initial robot positions and states

## Basic World Configuration

Let's start with a more complex world file that demonstrates advanced physics properties:

Create `~/digital_twin_ws/src/digital_twin_examples/worlds/physics_lab.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="physics_lab">
    <!-- Physics Engine Configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.0</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Lighting -->
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

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.0</restitution_coefficient>
              <threshold>100000</threshold>
            </bounce>
            <contact>
              <ode>
                <soft_cfm>0</soft_cfm>
                <soft_erp>0.2</soft_erp>
                <kp>1e+13</kp>
                <kd>1</kd>
                <max_vel>0.01</max_vel>
                <min_depth>0</min_depth>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>50 50</size>
            </plane>
          </geometry>
          <material>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Ramps with Different Angles -->
    <model name="ramp_15deg">
      <pose>-5 0 0 0 0.26 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 4 0.2</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 4 0.2</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="ramp_30deg">
      <pose>-2 0 0 0 0.52 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 4 0.2</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 4 0.2</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>2 1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.5</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>3 -1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.6 0.6 1.0</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.7</mu>
                <mu2>0.7</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.6 0.6 1.0</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Adjustable Gravity Test Area -->
    <model name="gravity_test_ball">
      <pose>-8 0 5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.1</mu>
                <mu2>0.1</mu2>
              </ode>
            </friction>
          </surface>
          <inertial>
            <mass>0.5</mass>
            <inertia>
              <ixx>0.001</ixx>
              <iyy>0.001</iyy>
              <izz>0.001</izz>
            </inertia>
          </inertial>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <material>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Physics Engine Configuration

The physics engine is the core of any simulation environment. Gazebo supports multiple physics engines:

### ODE (Open Dynamics Engine)
- Most commonly used in Gazebo
- Good balance of performance and accuracy
- Supports complex joint types and constraints

### Bullet Physics
- High-performance engine
- Good for complex collision detection
- Used in some specialized applications

### DART (Dynamic Animation and Robotics Toolkit)
- Advanced physics simulation
- Good for complex multi-body systems

## Gravity and Environmental Forces

Gravity is defined in the world file and affects all objects in the simulation:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
</physics>
```

You can modify gravity for different scenarios:
- Lunar gravity: `0 0 -1.6` (1/6th of Earth's gravity)
- Zero gravity: `0 0 0` (for space simulations)
- Custom gravity: Adjust values for experimental scenarios

## Collision Detection and Surfaces

Collision detection in Gazebo is configured through surface properties:

### Friction Parameters
- `mu` and `mu2`: Static and dynamic friction coefficients
- Higher values mean more resistance to sliding
- Values typically range from 0 (ice-like) to 1+ (high friction)

### Bounce Properties
- `restitution_coefficient`: How bouncy the surface is (0 = no bounce, 1 = perfectly elastic)
- `threshold`: Energy threshold for bounce behavior

### Contact Parameters
- `soft_erp`: Error reduction parameter (how quickly errors are corrected)
- `soft_cfm`: Constraint force mixing (stability vs. accuracy trade-off)
- `kp` and `kd`: Spring and damper coefficients for contact response

## Advanced World Features

### Terrain Generation
For more complex environments, you can create heightmap terrains:

```xml
<model name="terrain">
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### Wind Simulation
You can add environmental forces like wind:

```xml
<world name="windy_world">
  <physics type="ode">
    <gravity>0 0 -9.8</gravity>
  </physics>

  <model name="wind_generator">
    <static>true</static>
    <link name="link">
      <velocity_decay>0.0 0.0</velocity_decay>
      <acceleration>
        <x_msec2>0.1</x_msec2>
        <y_msec2>0.05</y_msec2>
        <z_msec2>0.0</z_msec2>
      </acceleration>
    </link>
  </model>
</world>
```

## Testing Physics Properties

Create a test script to verify physics behavior:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Point, Quaternion

class PhysicsTester(Node):
    def __init__(self):
        super().__init__('physics_tester')
        self.cli = self.create_client(GetEntityState, '/get_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /get_entity_state not available, waiting again...')
        self.req = GetEntityState.Request()

    def get_entity_state(self, entity_name, reference_frame=''):
        self.req.name = entity_name
        self.req.reference_frame = reference_frame

        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                position = response.state.pose.position
                self.get_logger().info(f'{entity_name} position: x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f}')
                return response.state
            else:
                self.get_logger().error(f'Failed to get state for {entity_name}: {response.status_message}')
                return None
        else:
            self.get_logger().error(f'Failed to call service for {entity_name}')
            return None

def main(args=None):
    rclpy.init(args=args)

    tester = PhysicsTester()

    # Test the gravity test ball to see how it falls
    for i in range(10):
        import time
        time.sleep(0.5)
        state = tester.get_entity_state('gravity_test_ball')
        if state:
            pos = state.pose.position
            if pos.z < 0.1:  # Ball has hit the ground
                print(f"Ball hit ground at time step {i}")
                break

    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for World Building

### Performance Optimization
- Keep collision meshes simple (use boxes/cylinders/spheres instead of complex meshes)
- Limit the number of dynamic objects in the scene
- Use appropriate physics step sizes (smaller steps = more accuracy but slower performance)

### Accuracy Considerations
- Match real-world physics parameters as closely as possible
- Test with known scenarios to validate physics behavior
- Use appropriate friction and restitution values for materials

### Debugging Tips
- Use Gazebo's visualization tools to see collision meshes
- Monitor simulation timing and real-time factor
- Check that gravity and other forces are correctly applied

## Summary

In this chapter, we've covered the essential aspects of creating physics-accurate simulation environments in Gazebo:
- Physics engine configuration and parameters
- Gravity and environmental force settings
- Collision detection and surface properties
- Advanced features like terrain and wind simulation
- Testing and validation techniques

These concepts form the foundation for creating realistic simulation environments that accurately represent real-world physics for robot testing and validation. In the next chapter, we'll focus on spawning and controlling humanoid robots in these environments.