---
title: Simulating Sensors - LiDAR, Depth Cameras, and IMUs
sidebar_position: 7
---

# Simulating Sensors: LiDAR, Depth Cameras, and IMUs

## Overview

In this chapter, we'll explore the simulation of critical robotic sensors in both Gazebo and Unity environments. Accurate sensor simulation is essential for developing and testing perception algorithms, navigation systems, and control strategies in a safe, repeatable environment. We'll cover LiDAR, depth cameras, and IMUs - three of the most important sensors for robotics applications.

## Understanding Sensor Simulation

Sensor simulation involves creating virtual sensors that generate data mimicking real-world sensors. This requires:

- **Physics-based simulation**: Accurate modeling of sensor physics (ray tracing, electromagnetic properties)
- **Noise modeling**: Realistic noise patterns that match real sensors
- **Timing accuracy**: Proper update rates and synchronization
- **Data format compatibility**: Output in standard ROS message formats

## LiDAR Simulation in Gazebo

### Creating LiDAR Sensors

LiDAR sensors in Gazebo are implemented using ray tracing. Here's an example URDF snippet with a LiDAR sensor:

```xml
<!-- Add to your robot URDF -->
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
</joint>

<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π -->
          <max_angle>3.14159</max_angle>    <!-- π -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Advanced LiDAR Configuration

For more sophisticated LiDAR simulation, consider these parameters:

```xml
<gazebo reference="lidar_link">
  <sensor name="advanced_lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>20</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1081</samples>  <!-- For 0.25° resolution -->
          <resolution>1</resolution>
          <min_angle>-2.35619</min_angle>  <!-- -135° -->
          <max_angle>2.35619</max_angle>   <!-- 135° -->
        </horizontal>
        <vertical>
          <samples>16</samples>
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle>  <!-- -15° -->
          <max_angle>0.2618</max_angle>   <!-- 15° -->
        </vertical>
      </scan>
      <range>
        <min>0.08</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="advanced_lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
      <gaussian_noise>0.005</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Processing in ROS

Create a LiDAR processing node to work with the simulated data:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from std_msgs.msg import Float32

class LiDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to LiDAR data
        self.subscription = self.create_subscription(
            LaserScan,
            '/robot/scan',
            self.lidar_callback,
            10
        )

        # Publishers for processed data
        self.obstacle_distance_pub = self.create_publisher(Float32, '/robot/obstacle_distance', 10)
        self.clear_path_pub = self.create_publisher(Float32, '/robot/clear_path', 10)

        # Parameters
        self.front_angle_range = 30  # degrees to check in front
        self.obstacle_threshold = 1.0  # meters

    def lidar_callback(self, msg):
        # Convert angle range to indices
        angle_increment = msg.angle_increment
        min_angle = msg.angle_min
        max_angle = msg.angle_max

        # Calculate front sector (between -30° and +30°)
        start_angle = -np.radians(self.front_angle_range / 2)
        end_angle = np.radians(self.front_angle_range / 2)

        start_idx = int((start_angle - min_angle) / angle_increment)
        end_idx = int((end_angle - min_angle) / angle_increment)

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(msg.ranges), end_idx)

        # Extract front range data
        front_ranges = msg.ranges[start_idx:end_idx]

        # Remove invalid ranges (inf, nan)
        valid_ranges = [r for r in front_ranges if r != float('inf') and not np.isnan(r)]

        if valid_ranges:
            min_distance = min(valid_ranges)

            # Publish obstacle distance
            obstacle_msg = Float32()
            obstacle_msg.data = min_distance
            self.obstacle_distance_pub.publish(obstacle_msg)

            # Determine if path is clear
            clear_msg = Float32()
            clear_msg.data = 1.0 if min_distance > self.obstacle_threshold else 0.0
            self.clear_path_pub.publish(clear_msg)

            self.get_logger().info(f'Front obstacle distance: {min_distance:.2f}m, Path clear: {clear_msg.data > 0.5}')
        else:
            self.get_logger().warn('No valid range data in front sector')

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LiDARProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass

    lidar_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Depth Camera Simulation in Gazebo

### Creating Depth Cameras

Depth cameras in Gazebo provide 3D point cloud data along with RGB images:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
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

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0.2" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <format>R8G8B8</format>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>rgb/image_raw:=rgb/image_raw</remapping>
        <remapping>depth/image_raw:=depth/image_raw</remapping>
        <remapping>depth/camera_info:=depth/camera_info</remapping>
      </ros>
      <frame_name>camera_link</frame_name>
      <baseline>0.2</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### Processing Depth Camera Data

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')

        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/robot/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/robot/depth/image_raw',
            self.depth_callback,
            10
        )

        # Store camera info
        self.camera_info = None

    def rgb_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the RGB image (example: detect objects)
            processed_image = self.process_rgb_image(cv_image)

            # Display the image
            cv2.imshow('RGB Camera', processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format
            if msg.encoding == '16UC1':
                dtype = np.uint16
            elif msg.encoding == '32FC1':
                dtype = np.float32
            else:
                self.get_logger().error(f'Unsupported depth image encoding: {msg.encoding}')
                return

            depth_image = np.frombuffer(msg.data, dtype=dtype)
            depth_image = depth_image.reshape((msg.height, msg.width))

            # Process depth image
            processed_depth = self.process_depth_image(depth_image)

            # Display depth image
            # Normalize for visualization
            depth_normalized = cv2.normalize(processed_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('Depth Camera', depth_normalized)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def process_rgb_image(self, image):
        # Example: Simple object detection
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (example)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def process_depth_image(self, depth_image):
        # Example: Distance to closest object in center region
        h, w = depth_image.shape
        center_h, center_w = h // 2, w // 2

        # Define center region (20% of image)
        region_size = min(h, w) // 5
        center_region = depth_image[
            center_h - region_size:center_h + region_size,
            center_w - region_size:center_w + region_size
        ]

        # Calculate statistics for center region
        valid_depths = center_region[np.isfinite(center_region) & (center_region > 0)]

        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            self.get_logger().info(f'Depth - Avg: {avg_depth:.2f}m, Min: {min_depth:.2f}m')

        return depth_image

def main(args=None):
    rclpy.init(args=args)
    depth_processor = DepthCameraProcessor()

    try:
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth Camera', cv2.WINDOW_AUTOSIZE)
        rclpy.spin(depth_processor)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    depth_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU Simulation in Gazebo

### Creating IMU Sensors

IMU sensors provide orientation, angular velocity, and linear acceleration data:

```xml
<link name="imu_link">
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
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

### Processing IMU Data

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
import numpy as np
import math

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.subscription = self.create_subscription(
            Imu,
            '/robot/imu',
            self.imu_callback,
            10
        )

        # Publishers for processed data
        self.roll_pub = self.create_publisher(Float32, '/robot/roll', 10)
        self.pitch_pub = self.create_publisher(Float32, '/robot/pitch', 10)
        self.yaw_pub = self.create_publisher(Float32, '/robot/yaw', 10)

        # Store previous values for velocity calculation
        self.prev_angular_velocity = Vector3()
        self.prev_time = self.get_clock().now()

    def imu_callback(self, msg):
        # Extract orientation from quaternion
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.w, orientation.x, orientation.y, orientation.z
        )

        # Publish Euler angles
        roll_msg = Float32()
        roll_msg.data = roll
        self.roll_pub.publish(roll_msg)

        pitch_msg = Float32()
        pitch_msg.data = pitch
        self.pitch_pub.publish(pitch_msg)

        yaw_msg = Float32()
        yaw_msg.data = yaw
        self.yaw_pub.publish(yaw_msg)

        # Calculate angular acceleration
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)
        dt = (current_time.nanoseconds - self.prev_time.nanoseconds) / 1e9

        if dt > 0:
            angular_velocity = msg.angular_velocity
            angular_acceleration_x = (angular_velocity.x - self.prev_angular_velocity.x) / dt
            angular_acceleration_y = (angular_velocity.y - self.prev_angular_velocity.y) / dt
            angular_acceleration_z = (angular_velocity.z - self.prev_angular_velocity.z) / dt

            self.get_logger().info(
                f'Orientation - Roll: {math.degrees(roll):.2f}°, '
                f'Pitch: {math.degrees(pitch):.2f}°, '
                f'Yaw: {math.degrees(yaw):.2f}°'
            )

        self.prev_angular_velocity = angular_velocity
        self.prev_time = current_time

    def quaternion_to_euler(self, w, x, y, z):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Using the ZYX intrinsic rotation convention
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    imu_processor = IMUProcessor()

    try:
        rclpy.spin(imu_processor)
    except KeyboardInterrupt:
        pass

    imu_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Simulation in Unity

### Unity Perception Package

Unity's Perception package provides high-quality sensor simulation for robotics and ML applications:

```csharp
// UnitySensorManager.cs
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Simulation;

public class UnitySensorManager : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public bool enableCamera = true;
    public bool enableLiDAR = true;
    public bool enableIMU = true;

    [Header("Camera Settings")]
    public Camera sensorCamera;
    public int cameraWidth = 640;
    public int cameraHeight = 480;
    public float cameraFov = 60f;

    [Header("LiDAR Settings")]
    public float lidarRange = 30f;
    public int lidarSamples = 360;
    public float lidarUpdateRate = 10f;

    private void Start()
    {
        SetupSensors();
    }

    void SetupSensors()
    {
        if (enableCamera && sensorCamera != null)
        {
            SetupCameraSensor();
        }

        if (enableLiDAR)
        {
            SetupLiDARSensor();
        }

        if (enableIMU)
        {
            SetupIMUMarker();
        }
    }

    void SetupCameraSensor()
    {
        // Configure the camera with perception capabilities
        sensorCamera.fieldOfView = cameraFov;
        sensorCamera.targetTexture = new RenderTexture(cameraWidth, cameraHeight, 24);

        // Add semantic segmentation if needed
        var segmentationManager = sensorCamera.gameObject.AddComponent<SemanticSegmentationLabeler>();
        segmentationManager.enabled = true;
    }

    void SetupLiDARSensor()
    {
        // Create a LiDAR sensor object
        GameObject lidarObject = new GameObject("LiDAR Sensor");
        lidarObject.transform.SetParent(transform);
        lidarObject.transform.localPosition = Vector3.zero;

        // Add LiDAR sensor component (this is conceptual - actual implementation may vary)
        // Unity's Perception package has various sensor implementations
        var lidarSensor = lidarObject.AddComponent<LidarSensor>();
        // Configure LiDAR parameters
    }

    void SetupIMUMarker()
    {
        // IMU data in Unity is typically derived from the object's transform changes
        // Unity doesn't have a direct IMU sensor, but we can calculate IMU-like data
        var imuMarker = gameObject.AddComponent<IMUMarker>();
        // Configure IMU parameters
    }
}
```

### Sensor Data Synchronization

To synchronize sensor data between Gazebo and Unity:

```csharp
// SensorSynchronizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

public class SensorSynchronizer : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosNamespace = "/robot";

    [Header("Sensor Topics")]
    public string lidarTopic = "/scan";
    public string cameraTopic = "/rgb/image_raw";
    public string imuTopic = "/imu";

    private ROSConnection ros;
    private Dictionary<string, float> sensorData = new Dictionary<string, float>();

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to sensor data
        ros.Subscribe<sensor_msgs_LaserScan>(rosNamespace + lidarTopic, OnLidarDataReceived);
        ros.Subscribe<sensor_msgs_Imu>(rosNamespace + imuTopic, OnIMUDataReceived);
    }

    void OnLidarDataReceived(sensor_msgs_LaserScan scan)
    {
        // Process LiDAR data
        if (scan.ranges.Count > 0)
        {
            float minRange = float.MaxValue;
            foreach (float range in scan.ranges)
            {
                if (range < minRange && range > scan.range_min && range < scan.range_max)
                {
                    minRange = range;
                }
            }

            if (minRange != float.MaxValue)
            {
                sensorData["min_obstacle_distance"] = minRange;
            }
        }
    }

    void OnIMUDataReceived(sensor_msgs_Imu imu)
    {
        // Process IMU data
        sensorData["orientation_x"] = (float)imu.orientation.x;
        sensorData["orientation_y"] = (float)imu.orientation.y;
        sensorData["orientation_z"] = (float)imu.orientation.z;
        sensorData["orientation_w"] = (float)imu.orientation.w;

        sensorData["angular_velocity_x"] = (float)imu.angular_velocity.x;
        sensorData["angular_velocity_y"] = (float)imu.angular_velocity.y;
        sensorData["angular_velocity_z"] = (float)imu.angular_velocity.z;
    }

    void Update()
    {
        // Use synchronized sensor data for visualization
        UpdateVisualization();
    }

    void UpdateVisualization()
    {
        // Update Unity visualization based on sensor data
        if (sensorData.ContainsKey("min_obstacle_distance"))
        {
            float distance = sensorData["min_obstacle_distance"];
            // Change color based on proximity
            Renderer renderer = GetComponent<Renderer>();
            if (renderer != null)
            {
                Color color = distance < 1.0f ? Color.red : Color.green;
                renderer.material.color = Color.Lerp(Color.green, Color.red,
                    Mathf.InverseLerp(5.0f, 0.5f, distance));
            }
        }
    }
}
```

## Multi-Sensor Fusion

### Combining Sensor Data

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import PointStamped
import numpy as np
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for all sensors
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/robot/scan',
            self.lidar_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/robot/imu',
            self.imu_callback,
            10
        )

        # Publisher for fused data
        self.obstacle_pub = self.create_publisher(PointStamped, '/robot/obstacle_position', 10)

        # Store sensor data
        self.imu_orientation = None
        self.lidar_data = None

    def lidar_callback(self, msg):
        self.lidar_data = msg
        self.fuse_sensor_data()

    def imu_callback(self, msg):
        self.imu_orientation = msg.orientation
        self.fuse_sensor_data()

    def fuse_sensor_data(self):
        if self.lidar_data is None or self.imu_orientation is None:
            return

        # Get the closest obstacle from LiDAR data
        ranges = self.lidar_data.ranges
        if len(ranges) == 0:
            return

        min_range_idx = np.argmin(ranges)
        min_range = ranges[min_range_idx]

        if min_range == float('inf') or np.isnan(min_range):
            return

        # Calculate angle of closest obstacle
        angle_increment = self.lidar_data.angle_increment
        angle = self.lidar_data.angle_min + (min_range_idx * angle_increment)

        # Convert polar to Cartesian coordinates (robot frame)
        x_robot = min_range * math.cos(angle)
        y_robot = min_range * math.sin(angle)

        # Transform to world frame using IMU orientation
        # (simplified - in practice, you'd use tf2 for proper transforms)
        roll, pitch, yaw = self.quaternion_to_euler(
            self.imu_orientation.w,
            self.imu_orientation.x,
            self.imu_orientation.y,
            self.imu_orientation.z
        )

        # Rotate the point by the robot's yaw
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x_world = x_robot * cos_yaw - y_robot * sin_yaw
        y_world = x_robot * sin_yaw + y_robot * cos_yaw

        # Create and publish obstacle position
        obstacle_msg = PointStamped()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = 'map'
        obstacle_msg.point.x = x_world
        obstacle_msg.point.y = y_world
        obstacle_msg.point.z = 0.0  # Assume ground level

        self.obstacle_pub.publish(obstacle_msg)

        self.get_logger().info(f'Fused obstacle position: ({x_world:.2f}, {y_world:.2f})')

    def quaternion_to_euler(self, w, x, y, z):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass

    fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Calibration and Validation

### Validation Techniques

1. **Ground Truth Comparison**: Compare sensor data with known ground truth
2. **Cross-Sensor Validation**: Verify consistency between different sensor types
3. **Temporal Consistency**: Check that sensor readings are consistent over time
4. **Environmental Validation**: Test sensors in various environmental conditions

### Performance Metrics

- **Accuracy**: How closely sensor data matches ground truth
- **Precision**: Consistency of repeated measurements
- **Latency**: Time delay between physical event and sensor reading
- **Update Rate**: Frequency of sensor data publication

## Troubleshooting Common Sensor Issues

### LiDAR Issues
- **Missing detections**: Check ray intersection settings and update rates
- **Range errors**: Verify min/max range parameters
- **Noise problems**: Adjust noise parameters to match real sensors

### Camera Issues
- **Image quality**: Check resolution and compression settings
- **Synchronization**: Ensure camera timing matches expectations
- **Distortion**: Apply proper distortion parameters

### IMU Issues
- **Drift**: Implement proper bias correction
- **Noise**: Use appropriate noise models
- **Alignment**: Verify sensor orientation in URDF

## Best Practices for Sensor Simulation

### Realism
- Use realistic noise models based on actual sensor specifications
- Implement proper timing and update rates
- Include sensor limitations and failure modes

### Performance
- Optimize sensor update rates for real-time performance
- Use appropriate sensor resolutions
- Implement efficient data processing pipelines

### Validation
- Regularly validate simulated sensors against real hardware
- Implement comprehensive testing procedures
- Document sensor characteristics and limitations

## Summary

In this chapter, we've covered the simulation of three critical robotic sensors:

- **LiDAR simulation**: Creating and processing 2D and 3D laser range data
- **Depth camera simulation**: Generating RGB and depth images with realistic noise
- **IMU simulation**: Providing orientation, angular velocity, and acceleration data
- **Sensor fusion**: Combining multiple sensor inputs for enhanced perception

Accurate sensor simulation is crucial for developing robust robotics applications that can transition from simulation to real hardware. The next chapter will compare Gazebo and Unity approaches and provide best practices for selecting the appropriate tool for specific applications.