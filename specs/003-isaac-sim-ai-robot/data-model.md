# Data Model: Isaac Sim and Isaac ROS Integration

## Entities

### Isaac Sim Environment
- **name**: String (unique identifier for the environment)
- **description**: String (human-readable description)
- **usd_file**: String (path to USD scene file)
- **simulation_settings**: Object (physics, rendering, and timing parameters)
- **robot_assets**: Array of RobotAsset (humanoid robots loaded in the environment)
- **sensor_configurations**: Array of SensorConfig (sensor settings for perception)
- **created_at**: DateTime
- **updated_at**: DateTime

**Validation rules**:
- name must be unique within the system
- usd_file must have .usd, .usda, or .usdc extension
- simulation_settings must include valid physics parameters

### Humanoid Robot Asset
- **name**: String (robot name, e.g., "ATLAS", "VALKYRIE", "Custom_Humanoid")
- **usd_file**: String (path to USD robot model file)
- **urdf_file**: String (path to URDF model file for ROS compatibility)
- **collision_meshes**: Array of String (paths to collision mesh files)
- **visual_meshes**: Array of String (paths to visual mesh files)
- **joints**: Array of Joint (kinematic chain definition)
- **links**: Array of Link (physical components)
- **sensors**: Array of Sensor (attached sensors)
- **base_position**: Vector3 (initial position in simulation)
- **base_orientation**: Quaternion (initial orientation in simulation)

**Validation rules**:
- name must be unique
- usd_file must be valid USD format
- base_position and base_orientation must be within simulation bounds

### Isaac ROS Package
- **name**: String (package name, e.g., "isaac_ros_visual_slam", "isaac_ros_point_cloud_localizer")
- **version**: String (version identifier, e.g., "3.2.0")
- **description**: String (functionality description)
- **dependencies**: Array of String (required packages)
- **parameters**: Object (configurable parameters)
- **topics**: Array of TopicInfo (ROS topics used by the package)
- **services**: Array of ServiceInfo (ROS services provided)
- **gpu_requirements**: GPURequirements (minimum GPU specifications)

**Validation rules**:
- name must follow ROS package naming conventions
- version must follow semantic versioning
- dependencies must be valid ROS packages

### GPU Requirements
- **cuda_compute_capability**: Float (minimum CUDA compute capability)
- **minimum_memory**: Integer (minimum GPU memory in MB)
- **recommended_memory**: Integer (recommended GPU memory in MB)
- **supported_architectures**: Array of String (e.g., "Ampere", "Ada Lovelace")
- **driver_version**: String (minimum NVIDIA driver version)

**Validation rules**:
- cuda_compute_capability must be >= 6.0
- minimum_memory must be > 0
- recommended_memory must be >= minimum_memory

### Sensor Configuration
- **name**: String (unique sensor identifier)
- **type**: Enum (camera, lidar, imu, depth_camera, fisheye_camera)
- **parent_link**: String (link to which sensor is attached)
- **position**: Vector3 (position relative to parent link)
- **orientation**: Quaternion (orientation relative to parent link)
- **parameters**: Object (sensor-specific parameters)
- **output_topics**: Array of String (ROS topics for sensor data)

**Validation rules**:
- sensor name must be unique within robot
- parent_link must exist in robot model
- parameters must match sensor type requirements

### Simulation Graph
- **name**: String (graph name)
- **description**: String (purpose and functionality)
- **nodes**: Array of GraphNode (processing nodes)
- **connections**: Array of GraphConnection (data flow connections)
- **input_interfaces**: Array of InterfaceSpec (external inputs)
- **output_interfaces**: Array of InterfaceSpec (external outputs)
- **performance_metrics**: PerformanceMetrics (timing and throughput data)

**Validation rules**:
- name must be unique
- nodes must have valid types
- connections must reference existing nodes

### Performance Metrics
- **processing_time**: Float (average processing time in ms)
- **frame_rate**: Float (frames per second)
- **memory_usage**: Integer (memory usage in MB)
- **gpu_utilization**: Float (GPU utilization percentage)
- **accuracy_metrics**: Object (perception accuracy measurements)

**Validation rules**:
- processing_time must be > 0
- frame_rate must be > 0
- memory_usage must be >= 0

### Bipedal Navigation Configuration
- **name**: String (configuration name)
- **robot_type**: String (type of humanoid robot)
- **nav2_params**: Object (Nav2 stack parameters)
- **footprint**: Polygon (robot collision footprint)
- **gait_patterns**: Array of GaitPattern (walking patterns)
- **terrain_adaptation**: TerrainAdaptation (terrain handling parameters)
- **safety_limits**: SafetyLimits (velocity, acceleration limits)

**Validation rules**:
- robot_type must match available robot models
- footprint must be a valid polygon
- gait_patterns must be kinematically valid

### Gait Pattern
- **name**: String (pattern name, e.g., "walk", "trot", "crawl")
- **joint_trajectories**: Array of JointTrajectory (joint movement sequences)
- **timing_parameters**: TimingParams (step duration, phase offsets)
- **stability_metrics**: StabilityMetrics (center of mass constraints)
- **energy_efficiency**: Float (relative energy consumption)

**Validation rules**:
- joint_trajectories must be kinematically valid
- timing_parameters must result in stable motion

## Relationships

1. Isaac Sim Environment **contains** multiple Humanoid Robot Assets
2. Isaac Sim Environment **uses** multiple Sensor Configurations
3. Isaac ROS Package **has** GPU Requirements
4. Humanoid Robot Asset **has** multiple Joints and Links
5. Humanoid Robot Asset **has** multiple Sensors with configurations
6. Simulation Graph **contains** multiple Graph Nodes
7. Bipedal Navigation Configuration **uses** multiple Gait Patterns
8. Performance Metrics **measures** Isaac ROS Package execution

## State Transitions

### Isaac Sim Environment States
- **Created**: Environment definition exists but not loaded
- **Loaded**: Environment is loaded in Isaac Sim
- **Running**: Simulation is actively executing
- **Paused**: Simulation is temporarily stopped
- **Stopped**: Simulation has been terminated

### Isaac ROS Package States
- **Initialized**: Package is loaded but not processing
- **Running**: Package is actively processing data
- **Paused**: Processing is temporarily suspended
- **Error**: Package encountered an error
- **Shutdown**: Package has been properly terminated

### Humanoid Robot States
- **Idle**: Robot is loaded but not active in simulation
- **Spawned**: Robot has been placed in simulation environment
- **Active**: Robot is receiving control commands
- **Paused**: Robot simulation is paused
- **Terminated**: Robot simulation has ended