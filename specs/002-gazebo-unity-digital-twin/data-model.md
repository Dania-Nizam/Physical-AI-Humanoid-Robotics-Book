# Data Model: Gazebo & Unity Digital Twin

## Entities

### Digital Twin Environment
- **name**: String (unique identifier for the environment)
- **description**: String (human-readable description)
- **gazebo_world_file**: String (path to SDF world file)
- **unity_scene_file**: String (path to Unity scene file)
- **physics_properties**: Object (gravity, friction, damping settings)
- **lighting_settings**: Object (ambient light, directional light properties)
- **environment_objects**: Array of EnvironmentObject
- **created_at**: DateTime
- **updated_at**: DateTime

**Validation rules**:
- name must be unique within the system
- gazebo_world_file must have .sdf or .world extension
- unity_scene_file must have .unity extension

### Humanoid Robot Model
- **name**: String (robot name, e.g., "Atlas", "HRP-2", "Custom_Biped")
- **urdf_file**: String (path to URDF model file)
- **sdf_file**: String (path to SDF model file for Gazebo)
- **collision_meshes**: Array of String (paths to collision mesh files)
- **visual_meshes**: Array of String (paths to visual mesh files)
- **joints**: Array of Joint
- **links**: Array of Link
- **sensors**: Array of Sensor
- **base_position**: Vector3 (initial position in simulation)
- **base_orientation**: Quaternion (initial orientation in simulation)

**Validation rules**:
- name must be unique
- urdf_file must be valid URDF format
- base_position and base_orientation must be within simulation bounds

### Joint
- **name**: String (unique within robot)
- **type**: Enum (revolute, prismatic, fixed, continuous, floating, planar)
- **parent_link**: String (name of parent link)
- **child_link**: String (name of child link)
- **axis**: Vector3 (joint axis)
- **limits**: JointLimits (min/max position, velocity, effort)
- **dynamics**: JointDynamics (damping, friction)

**Validation rules**:
- joint name must be unique within robot
- parent_link and child_link must exist in robot model
- axis must be normalized vector

### Link
- **name**: String (unique within robot)
- **inertial**: Inertial (mass, center of mass, inertia tensor)
- **visual**: Visual (geometry, material, origin)
- **collision**: Collision (geometry, surface properties)

**Validation rules**:
- link name must be unique within robot
- mass must be positive
- geometry must be valid (box, cylinder, sphere, mesh)

### Sensor
- **name**: String (unique within robot)
- **type**: Enum (lidar, depth_camera, imu, camera, gps, force_torque)
- **parent_link**: String (link to which sensor is attached)
- **position**: Vector3 (position relative to parent link)
- **orientation**: Quaternion (orientation relative to parent link)
- **parameters**: Object (sensor-specific parameters)

**Validation rules**:
- sensor name must be unique within robot
- parent_link must exist in robot model
- parameters must match sensor type requirements

### EnvironmentObject
- **name**: String (object name)
- **type**: Enum (static, dynamic, kinematic)
- **geometry**: Object (shape, dimensions, mesh)
- **position**: Vector3 (world position)
- **orientation**: Quaternion (world orientation)
- **properties**: Object (mass, friction, restitution for physics)

**Validation rules**:
- name must be unique within environment
- geometry must be valid
- position must be within environment bounds

### SimulationState
- **environment_id**: String (reference to DigitalTwinEnvironment)
- **robot_states**: Array of RobotState
- **timestamp**: DateTime
- **simulation_time**: Float (simulation time in seconds)
- **real_time_factor**: Float (real-time factor of simulation)

### RobotState
- **robot_id**: String (reference to Humanoid Robot Model)
- **joint_positions**: Object (mapping of joint names to positions)
- **joint_velocities**: Object (mapping of joint names to velocities)
- **joint_efforts**: Object (mapping of joint names to efforts)
- **base_position**: Vector3 (robot base position in world)
- **base_orientation**: Quaternion (robot base orientation in world)
- **sensor_data**: Object (mapping of sensor names to current readings)

## Relationships

1. Digital Twin Environment **contains** multiple Environment Objects
2. Digital Twin Environment **supports** multiple Humanoid Robot Models
3. Humanoid Robot Model **has** multiple Joints
4. Humanoid Robot Model **has** multiple Links
5. Humanoid Robot Model **has** multiple Sensors
6. Joint **connects** two Links
7. Link **has** one Inertial component
8. Link **has** one Visual component
9. Link **has** one Collision component
10. SimulationState **contains** multiple RobotStates

## State Transitions

### Humanoid Robot Model States
- **Idle**: Robot is loaded but not active in simulation
- **Spawned**: Robot has been placed in simulation environment
- **Active**: Robot is receiving control commands
- **Paused**: Robot simulation is paused
- **Terminated**: Robot simulation has ended

### Simulation Environment States
- **Created**: Environment definition exists but not loaded
- **Loaded**: Environment is loaded in Gazebo/Unity
- **Running**: Simulation is actively executing
- **Paused**: Simulation is temporarily stopped
- **Stopped**: Simulation has been terminated