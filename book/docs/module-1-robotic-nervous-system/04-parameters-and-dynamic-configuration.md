---
title: Parameters and Dynamic Configuration
description: Managing configuration values in ROS 2 nodes at runtime
tags: [ros2, parameters, configuration, dynamic]
wordCount: 1100
---

# Parameters and Dynamic Configuration

## Learning Objectives

By the end of this chapter, you will be able to:
- Define parameters in ROS 2 and understand their purpose
- Declare and use parameters in Python nodes
- Configure parameters at launch time and runtime
- Implement parameter callbacks for dynamic reconfiguration
- Understand parameter namespaces and best practices

## Understanding Parameters

Parameters in ROS 2 are configuration values that can be set at runtime and shared between nodes. They provide a way to configure node behavior without recompiling or restarting nodes. Parameters can be:

- **Primitive types**: integers, floats, strings, booleans
- **Complex types**: lists, dictionaries
- **Dynamic**: changed during runtime

Parameters are particularly useful for:
- Tuning algorithm parameters (PID gains, thresholds)
- Setting operational modes
- Configuring hardware interfaces
- Specifying file paths and URLs

## Parameter Declaration and Usage

In ROS 2, parameters must be declared before they can be used. This allows for type checking and provides default values.

### Declaring Parameters in Python

```python
import rclpy
from rclpy.node import Node

class ParameterExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_example_node')

        # Declare parameters with default values
        self.declare_parameter('motor_speed', 1.0)
        self.declare_parameter('sensor_threshold', 0.5)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('robot_name', 'default_robot')

        # Access parameter values
        self.motor_speed = self.get_parameter('motor_speed').value
        self.sensor_threshold = self.get_parameter('sensor_threshold').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(f'Initialized with motor_speed: {self.motor_speed}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterExampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameter Callbacks

Parameter callbacks allow nodes to react to parameter changes at runtime:

```python
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

class ParameterCallbackNode(Node):
    def __init__(self):
        super().__init__('parameter_callback_node')

        self.declare_parameter('threshold', 1.0)
        self.threshold = self.get_parameter('threshold').value

        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.threshold = param.value
                self.get_logger().info(f'Threshold updated to: {self.threshold}')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterCallbackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameter Namespaces

Parameters can be organized using namespaces to avoid naming conflicts:

```python
# In your node
self.declare_parameter('sensors.lidar.range_max', 10.0)
self.declare_parameter('sensors.camera.resolution_width', 640)

# Access with full name
lidar_range = self.get_parameter('sensors.lidar.range_max').value
```

## Launch File Configuration

Parameters can be set in launch files:

```python
# example_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'motor_speed': 2.5},
                {'sensor_threshold': 0.8},
                {'debug_mode': True}
            ]
        )
    ])
```

## YAML Parameter Files

Parameters can also be loaded from YAML files:

```yaml
# params.yaml
my_node:
  ros__parameters:
    motor_speed: 2.5
    sensor_threshold: 0.8
    debug_mode: true
    robot_name: "my_robot"
```

Loading in launch file:
```python
Node(
    package='my_package',
    executable='my_node',
    name='my_node',
    parameters=['path/to/params.yaml']
)
```

## Command-Line Parameter Management

ROS 2 provides command-line tools for parameter management:

- `ros2 param list`: List all parameters of a node
- `ros2 param get <node_name> <param_name>`: Get a parameter value
- `ros2 param set <node_name> <param_name> <value>`: Set a parameter value
- `ros2 param dump <node_name>`: Dump all parameters to a file

Example:
```bash
# Get parameter
ros2 param get /my_node motor_speed

# Set parameter at runtime
ros2 param set /my_node sensor_threshold 0.75

# List all parameters for a node
ros2 param list /my_node
```

## Parameter Descriptors

You can add metadata to parameters using descriptors:

```python
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import FloatingPointRange

class ParameterDescriptorNode(Node):
    def __init__(self):
        super().__init__('parameter_descriptor_node')

        # Create a descriptor with constraints
        descriptor = ParameterDescriptor(
            description='Motor speed in radians per second',
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0)]
        )

        # Declare parameter with descriptor
        self.declare_parameter('motor_speed', 1.0, descriptor)
```

## Best Practices for Parameters

1. **Use descriptive names**: Choose clear, meaningful parameter names
2. **Provide defaults**: Always provide sensible default values
3. **Validate values**: Implement validation in parameter callbacks
4. **Document parameters**: Document parameter purpose and valid ranges
5. **Use appropriate types**: Choose the right data type for your parameters
6. **Group related parameters**: Use namespaces for related parameters

## Dynamic Reconfiguration

Parameters enable dynamic reconfiguration of nodes without restart. This is particularly useful for:

- Tuning control parameters during operation
- Switching operational modes
- Adjusting sensitivity settings
- Changing operational parameters based on conditions

## Parameter Limitations

While parameters are powerful, they have some limitations:

- Not suitable for high-frequency updates (use topics for continuous data)
- Should not be used for large data structures
- Changes are not guaranteed to be atomic across multiple parameters

## Summary

Parameters provide a flexible way to configure ROS 2 nodes at runtime. They enable dynamic reconfiguration without requiring node restarts, making them ideal for tuning operational parameters and switching between different operational modes. Understanding parameter declaration, usage, and management is essential for creating flexible and configurable ROS 2 systems.

In the next chapter, we'll explore launch files and how to compose complex robotic systems.