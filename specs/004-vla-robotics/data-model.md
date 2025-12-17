# Data Model: Vision-Language-Action (VLA) Pipeline

## Entities

### Voice Command
- **transcription**: String (converted text from speech)
- **confidence**: Float (speech recognition confidence level, 0.0-1.0)
- **timestamp**: DateTime (when the command was issued)
- **language**: String (detected language of the command)
- **raw_audio**: AudioData (original audio data, optional for simulation)
- **status**: Enum (pending, processed, failed)

**Validation rules**:
- transcription must not be empty
- confidence must be between 0.0 and 1.0
- timestamp must be within current session

### LLM Response
- **command**: String (interpreted command from LLM)
- **action_sequence**: Array of ActionStep (structured sequence of actions)
- **intent**: String (recognized intent from natural language)
- **entities**: Array of String (extracted named entities from command)
- **confidence**: Float (LLM confidence in interpretation, 0.0-1.0)
- **timestamp**: DateTime (when response was generated)

**Validation rules**:
- command must not be empty
- action_sequence must contain at least one step
- confidence must be between 0.0 and 1.0

### ActionStep
- **action_type**: Enum (navigation, manipulation, perception, communication)
- **parameters**: Object (action-specific parameters)
- **target_object**: String (optional target object name)
- **target_location**: Vector3 (optional target location coordinates)
- **timeout**: Integer (maximum time to complete action in seconds)
- **retry_count**: Integer (number of retry attempts allowed)

**Validation rules**:
- action_type must be a valid enum value
- parameters must match the expected structure for the action type
- timeout must be positive

### RobotState
- **position**: Vector3 (current robot position in simulation)
- **orientation**: Quaternion (current robot orientation)
- **velocity**: Vector3 (current robot velocity)
- **joint_angles**: Array of Float (current joint angles for humanoid robot)
- **gripper_state**: Enum (open, closed, partially_open)
- **battery_level**: Float (remaining battery percentage, 0.0-1.0)
- **last_action_status**: Enum (success, failed, in_progress)

**Validation rules**:
- position coordinates must be within simulation bounds
- joint_angles array length must match robot DOF
- battery_level must be between 0.0 and 1.0

### PerceptionData
- **detected_objects**: Array of DetectedObject (objects detected in environment)
- **environment_map**: OccupancyGrid (2D/3D map of environment)
- **camera_feed**: ImageData (current camera feed)
- **lidar_data**: PointCloud (LiDAR sensor data)
- **timestamp**: DateTime (when data was captured)
- **confidence_scores**: Array of Float (confidence for each detection)

**Validation rules**:
- detected_objects array elements must have valid object properties
- timestamp must be recent (within perception window)

### DetectedObject
- **name**: String (object identifier)
- **class**: String (object category/type)
- **position**: Vector3 (object position in world coordinates)
- **bounding_box**: BoundingBox3D (3D bounding box around object)
- **confidence**: Float (detection confidence, 0.0-1.0)
- **is_graspable**: Boolean (whether object can be grasped)

**Validation rules**:
- name must be unique in current scene
- position coordinates must be within simulation bounds
- confidence must be between 0.0 and 1.0

### BoundingBox3D
- **center**: Vector3 (center of bounding box)
- **dimensions**: Vector3 (width, height, depth)
- **rotation**: Quaternion (rotation of bounding box)
- **corners**: Array of Vector3 (8 corner points of box)

**Validation rules**:
- dimensions must be positive values
- corners array must contain exactly 8 points

### SimulationEnvironment
- **name**: String (environment identifier)
- **description**: String (environment description)
- **objects**: Array of SimulatedObject (objects in environment)
- **navigation_areas**: Array of Polygon (navigable areas)
- **obstacles**: Array of Obstacle (obstacles in environment)
- **spawn_points**: Array of Vector3 (valid robot spawn locations)

**Validation rules**:
- name must be unique
- navigation_areas must not overlap
- spawn_points must be in navigable areas

### SimulatedObject
- **name**: String (object name)
- **type**: Enum (furniture, decoration, interactive, graspable)
- **position**: Vector3 (object position)
- **orientation**: Quaternion (object orientation)
- **physics_properties**: PhysicsProperties (mass, friction, etc.)
- **collidable**: Boolean (whether object participates in physics simulation)

**Validation rules**:
- name must be unique in environment
- position must be within environment bounds

### PhysicsProperties
- **mass**: Float (object mass in kg)
- **friction**: Float (surface friction coefficient)
- **restitution**: Float (bounciness, 0.0-1.0)
- **density**: Float (material density)
- **is_static**: Boolean (fixed in place if true)

**Validation rules**:
- mass must be positive
- friction and restitution must be between 0.0 and 1.0

### NavigationGoal
- **target_location**: Vector3 (destination coordinates)
- **target_orientation**: Quaternion (desired final orientation)
- **waypoints**: Array of Vector3 (intermediate waypoints)
- **path_constraints**: PathConstraints (navigation constraints)
- **completion_criteria**: String (conditions for successful navigation)

**Validation rules**:
- target_location must be in navigable area
- waypoints must form a connected path

### PathConstraints
- **max_velocity**: Float (maximum allowed velocity)
- **min_distance_to_obstacles**: Float (safety distance)
- **preferred_surface**: String (preferred terrain type)
- **avoid_dynamic_objects**: Boolean (whether to avoid moving objects)

**Validation rules**:
- max_velocity must be positive
- min_distance_to_obstacles must be non-negative

## Relationships

1. Voice Command **generates** LLM Response
2. LLM Response **contains** multiple ActionStep
3. ActionStep **operates on** RobotState
4. RobotState **perceives** PerceptionData
5. PerceptionData **detects** multiple DetectedObject
6. DetectedObject **exists in** SimulationEnvironment
7. NavigationGoal **targets** SimulationEnvironment
8. SimulatedObject **has** PhysicsProperties
9. NavigationGoal **follows** PathConstraints

## State Transitions

### Voice Command States
- **Pending**: Command received but not yet processed
- **Processing**: Speech recognition in progress
- **Processed**: Text transcription completed
- **Failed**: Speech recognition failed

### Action Step States
- **Queued**: Action scheduled but not started
- **Executing**: Action currently running
- **Completed**: Action finished successfully
- **Failed**: Action failed to complete
- **Cancelled**: Action interrupted or cancelled

### Robot State Transitions
- **Idle**: Robot stationary and awaiting commands
- **Navigating**: Moving to target location
- **Perceiving**: Performing perception tasks
- **Manipulating**: Performing manipulation tasks
- **Communicating**: Responding to user or providing status
- **Error**: Robot in error state requiring intervention

### Simulation Environment States
- **Loading**: Environment assets being loaded
- **Ready**: Environment loaded and ready for simulation
- **Active**: Simulation currently running
- **Paused**: Simulation temporarily stopped
- **Resetting**: Environment being reset to initial state