---
title: "Synthetic Data Generation for Perception Training"
sidebar_position: 4
---

# Synthetic Data Generation for Perception Training

## Overview

Synthetic data generation is a cornerstone of modern robotics and AI development. In the context of NVIDIA Isaac Sim, synthetic data generation enables the creation of large, diverse, and perfectly labeled datasets that can be used to train perception models without the need for expensive real-world data collection.

By the end of this chapter, you will be able to:
- Understand the principles of synthetic data generation in Isaac Sim
- Configure synthetic data pipelines for perception training
- Generate labeled datasets for computer vision tasks
- Optimize synthetic data quality for real-world transfer

## Introduction to Synthetic Data Generation

Synthetic data generation is a cornerstone of modern robotics and AI development. In the context of NVIDIA Isaac Sim, synthetic data generation enables the creation of large, diverse, and perfectly labeled datasets that can be used to train perception models without the need for expensive real-world data collection.

The advantages of synthetic data include:
- **Perfect annotations**: Ground truth labels for segmentation, bounding boxes, and 3D poses
- **Infinite variations**: Control over lighting, weather, backgrounds, and object configurations
- **Safety**: Train on dangerous scenarios without real-world risk
- **Cost efficiency**: Generate thousands of examples without physical hardware

## Isaac Sim Synthetic Data Pipeline

Isaac Sim provides a comprehensive synthetic data generation pipeline through its Synthetic Data Extension. This extension allows you to:

1. **Define data generation scenarios** with specific parameters
2. **Configure sensors** to capture multiple modalities (RGB, depth, semantic segmentation)
3. **Generate annotations** automatically (2D/3D bounding boxes, segmentation masks, keypoint labels)
4. **Export data** in standard formats for ML training frameworks

### Setting up the Synthetic Data Extension

First, ensure the Synthetic Data Extension is enabled in Isaac Sim:

```python
import omni.syntheticdata as sd

# Enable the synthetic data extension
sd.acquire_syntheticdata_interface().set_camera_path("/World/Camera")
```

### Configuring Data Generation Parameters

Synthetic data generation in Isaac Sim involves configuring several key parameters:

```python
# Configure synthetic data settings
from omni.isaac.synthetic_utils import SyntheticDataHelper

synthetic_helper = SyntheticDataHelper()
synthetic_helper.set_output_directory("/path/to/output/directory")
synthetic_helper.set_dataset_size(10000)  # Number of frames to generate
synthetic_helper.set_scene_variations({
    "lighting": ["sunny", "cloudy", "night"],
    "weather": ["clear", "rainy", "foggy"],
    "objects": ["car", "pedestrian", "bicycle"]
})
```

## Generating Labeled Datasets

### Semantic Segmentation Data

Semantic segmentation is crucial for scene understanding in robotics. Isaac Sim can generate pixel-perfect segmentation masks:

```python
# Configure semantic segmentation data generation
from omni.isaac.synthetic_utils import visualize_segmentation

# Assign semantic labels to objects in the scene
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage

# Add semantic data to a cube
add_semantic_data_to_stage(
    prim_path="/World/Cube",
    semantic_label="obstacle",
    type_label="static"
)

# Generate segmentation data
visualize_segmentation.capture_segmentation_data()
```

### 3D Bounding Box Annotations

For object detection tasks, Isaac Sim can generate 3D bounding box annotations:

```python
# Generate 3D bounding box data
from omni.syntheticdata import bindings as sd

# Capture 3D bounding box information
bounding_box_3d = sd.capture_3d_bounding_box_data()
print(f"Object bounding boxes: {bounding_box_3d}")
```

### Depth and Normal Maps

Depth and surface normal maps are essential for 3D understanding:

```python
# Capture depth and normal maps
from omni.syntheticdata import capture_depth_data, capture_normal_data

depth_map = capture_depth_data()
normal_map = capture_normal_data()

# Process the captured data
print(f"Depth map shape: {depth_map.shape}")
print(f"Normal map shape: {normal_map.shape}")
```

## Creating a Synthetic Data Generation Script

Let's create a complete script for generating synthetic data:

```python
#!/usr/bin/env python3
"""
Synthetic Data Generation Script for Isaac Sim
Generates labeled datasets for perception training
"""

import omni
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage
from omni.syntheticdata import capture_instance_segmentation, capture_bounding_box_2d_tight

class SyntheticDataGenerator:
    def __init__(self, output_dir="/workspace/synthetic_data"):
        self.output_dir = output_dir
        self.world = World(stage_units_in_meters=1.0)
        self.synthetic_helper = SyntheticDataHelper()

    def setup_scene(self):
        """Setup the scene with various objects for data generation"""
        # Add a ground plane
        self.world.scene.add_ground_plane("/World/Ground", static_friction=0.1, dynamic_friction=0.1, restitution=0.0)

        # Add various objects with semantic labels
        from omni.isaac.core.objects import DynamicCuboid

        # Add labeled objects
        cube1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube1",
                name="cube1",
                position=np.array([1.0, 0.0, 0.5]),
                size=np.array([0.5, 0.5, 0.5]),
                color=np.array([0.8, 0.1, 0.1])
            )
        )

        cube2 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube2",
                name="cube2",
                position=np.array([-1.0, 0.0, 0.5]),
                size=np.array([0.3, 0.3, 0.3]),
                color=np.array([0.1, 0.8, 0.1])
            )
        )

        # Add semantic labels
        add_semantic_data_to_stage("/World/Cube1", semantic_label="red_cube", type_label="obstacle")
        add_semantic_data_to_stage("/World/Cube2", semantic_label="green_cube", type_label="obstacle")

        # Add a camera for data capture
        from omni.isaac.sensor import Camera
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([2.0, 2.0, 2.0]),
            look_at_target=np.array([0, 0, 0.5])
        )
        self.world.scene.add(self.camera)

    def generate_dataset(self, num_frames=100):
        """Generate synthetic dataset with multiple modalities"""
        self.world.reset()

        for frame_idx in range(num_frames):
            # Randomize scene slightly
            self.randomize_scene(frame_idx)

            # Step the physics
            self.world.step(render=True)

            # Capture multiple modalities
            self.capture_frame(frame_idx)

            carb.log_info(f"Captured frame {frame_idx + 1}/{num_frames}")

    def randomize_scene(self, frame_idx):
        """Randomize scene parameters for variation"""
        # Randomize lighting
        light_prim = get_prim_at_path("/World/Light")
        if light_prim:
            # Apply random light intensity or color variation
            pass

        # Randomize object positions slightly
        cube1 = self.world.scene.get_object("cube1")
        cube2 = self.world.scene.get_object("cube2")

        if cube1 and cube2:
            import random
            # Add small random offsets to positions
            new_pos1 = [1.0 + random.uniform(-0.1, 0.1), 0.0 + random.uniform(-0.1, 0.1), 0.5]
            new_pos2 = [-1.0 + random.uniform(-0.1, 0.1), 0.0 + random.uniform(-0.1, 0.1), 0.5]

            cube1.set_world_pose(position=np.array(new_pos1))
            cube2.set_world_pose(position=np.array(new_pos2))

    def capture_frame(self, frame_idx):
        """Capture all modalities for a single frame"""
        # Capture RGB image
        rgb_data = self.camera.get_rgb()

        # Capture depth
        depth_data = self.camera.get_depth()

        # Capture semantic segmentation
        semantic_data = self.camera.get_semantic_segmentation()

        # Capture bounding boxes
        bbox_data = self.camera.get_bounding_boxes_2d_tight()

        # Save the data
        self.save_frame_data(frame_idx, rgb_data, depth_data, semantic_data, bbox_data)

    def save_frame_data(self, frame_idx, rgb, depth, semantic, bbox):
        """Save captured data to disk"""
        import os
        from PIL import Image
        import json

        frame_dir = os.path.join(self.output_dir, f"frame_{frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Save RGB image
        rgb_image = Image.fromarray((rgb * 255).astype(np.uint8))
        rgb_image.save(os.path.join(frame_dir, "rgb.png"))

        # Save depth as numpy array
        np.save(os.path.join(frame_dir, "depth.npy"), depth)

        # Save semantic segmentation
        semantic_image = Image.fromarray(semantic.astype(np.uint16))
        semantic_image.save(os.path.join(frame_dir, "semantic.png"))

        # Save bounding box annotations
        bbox_dict = {
            "frame": frame_idx,
            "objects": []
        }

        for bbox_info in bbox:
            obj_info = {
                "label": bbox_info["label"],
                "bbox": [int(coord) for coord in bbox_info["bbox"]],
                "instance_id": bbox_info["instance_id"]
            }
            bbox_dict["objects"].append(obj_info)

        with open(os.path.join(frame_dir, "annotations.json"), "w") as f:
            json.dump(bbox_dict, f, indent=2)

    def run(self):
        """Run the complete synthetic data generation pipeline"""
        carb.log_info("Setting up scene...")
        self.setup_scene()

        carb.log_info("Starting data generation...")
        self.generate_dataset(num_frames=100)

        carb.log_info("Data generation completed!")

def main():
    """Main function to run the synthetic data generator"""
    generator = SyntheticDataGenerator()
    generator.run()

if __name__ == "__main__":
    main()
```

## Code Examples

### Synthetic Data Generation Pipeline

```python
#!/usr/bin/env python3
"""
Complete Synthetic Data Generation Pipeline Example
This script demonstrates a complete synthetic data generation workflow
"""

import omni
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core.utils.semantics import add_semantic_data_to_stage
from omni.syntheticdata import capture_instance_segmentation, capture_bounding_box_2d_tight
import os
from PIL import Image
import json

class CompleteSyntheticDataPipeline:
    def __init__(self, output_dir="/workspace/synthetic_data"):
        self.output_dir = output_dir
        self.world = World(stage_units_in_meters=1.0)
        self.synthetic_helper = SyntheticDataHelper()

    def setup_complete_scene(self):
        """Setup a complete scene with multiple objects for data generation"""
        # Add a ground plane
        self.world.scene.add_ground_plane("/World/Ground", static_friction=0.1, dynamic_friction=0.1, restitution=0.0)

        # Add various objects with semantic labels
        from omni.isaac.core.objects import DynamicCuboid

        # Add labeled objects
        cube1 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube1",
                name="cube1",
                position=np.array([1.0, 0.0, 0.5]),
                size=np.array([0.5, 0.5, 0.5]),
                color=np.array([0.8, 0.1, 0.1])
            )
        )

        cube2 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube2",
                name="cube2",
                position=np.array([-1.0, 0.0, 0.5]),
                size=np.array([0.3, 0.3, 0.3]),
                color=np.array([0.1, 0.8, 0.1])
            )
        )

        # Add semantic labels
        add_semantic_data_to_stage("/World/Cube1", semantic_label="red_cube", type_label="obstacle")
        add_semantic_data_to_stage("/World/Cube2", semantic_label="green_cube", type_label="obstacle")

        # Add a camera for data capture
        from omni.isaac.sensor import Camera
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([2.0, 2.0, 2.0]),
            look_at_target=np.array([0, 0, 0.5])
        )
        self.world.scene.add(self.camera)

    def run_complete_pipeline(self):
        """Run the complete synthetic data generation pipeline"""
        self.world.reset()

        for frame_idx in range(50):  # Generate 50 frames
            # Randomize scene slightly
            self.randomize_scene(frame_idx)

            # Step the physics
            self.world.step(render=True)

            # Capture multiple modalities
            self.capture_frame(frame_idx)

            carb.log_info(f"Captured frame {frame_idx + 1}/50")

    def randomize_scene(self, frame_idx):
        """Randomize scene parameters for variation"""
        # Randomize lighting
        light_prim = get_prim_at_path("/World/Light")
        if light_prim:
            # Apply random light intensity or color variation
            pass

        # Randomize object positions slightly
        cube1 = self.world.scene.get_object("cube1")
        cube2 = self.world.scene.get_object("cube2")

        if cube1 and cube2:
            import random
            # Add small random offsets to positions
            new_pos1 = [1.0 + random.uniform(-0.1, 0.1), 0.0 + random.uniform(-0.1, 0.1), 0.5]
            new_pos2 = [-1.0 + random.uniform(-0.1, 0.1), 0.0 + random.uniform(-0.1, 0.1), 0.5]

            cube1.set_world_pose(position=np.array(new_pos1))
            cube2.set_world_pose(position=np.array(new_pos2))

    def capture_frame(self, frame_idx):
        """Capture all modalities for a single frame"""
        # Capture RGB image
        rgb_data = self.camera.get_rgb()

        # Capture depth
        depth_data = self.camera.get_depth()

        # Capture semantic segmentation
        semantic_data = self.camera.get_semantic_segmentation()

        # Capture bounding boxes
        bbox_data = self.camera.get_bounding_boxes_2d_tight()

        # Save the data
        self.save_frame_data(frame_idx, rgb_data, depth_data, semantic_data, bbox_data)

    def save_frame_data(self, frame_idx, rgb, depth, semantic, bbox):
        """Save captured data to disk"""
        frame_dir = os.path.join(self.output_dir, f"frame_{frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Save RGB image
        rgb_image = Image.fromarray((rgb * 255).astype(np.uint8))
        rgb_image.save(os.path.join(frame_dir, "rgb.png"))

        # Save depth as numpy array
        np.save(os.path.join(frame_dir, "depth.npy"), depth)

        # Save semantic segmentation
        semantic_image = Image.fromarray(semantic.astype(np.uint16))
        semantic_image.save(os.path.join(frame_dir, "semantic.png"))

        # Save bounding box annotations
        bbox_dict = {
            "frame": frame_idx,
            "objects": []
        }

        for bbox_info in bbox:
            obj_info = {
                "label": bbox_info["label"],
                "bbox": [int(coord) for coord in bbox_info["bbox"]],
                "instance_id": bbox_info["instance_id"]
            }
            bbox_dict["objects"].append(obj_info)

        with open(os.path.join(frame_dir, "annotations.json"), "w") as f:
            json.dump(bbox_dict, f, indent=2)

def main():
    """Main function to run the complete synthetic data pipeline"""
    pipeline = CompleteSyntheticDataPipeline()
    pipeline.setup_complete_scene()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()
```

## Best Practices

### Synthetic Data Generation Best Practices

1. **Scene Variation**: Introduce sufficient variation in lighting, camera angles, and object positions to ensure model robustness
2. **Annotation Quality**: Verify that all annotations are accurate and complete
3. **Dataset Balance**: Ensure balanced representation of different classes and scenarios
4. **Validation**: Implement validation checks to ensure data quality
5. **Performance**: Optimize generation pipeline for efficiency while maintaining quality

## Summary

This chapter covered the fundamentals of synthetic data generation in NVIDIA Isaac Sim for perception training. We explored the synthetic data pipeline, configuration of data generation parameters, and the process of creating labeled datasets for machine learning applications. You now understand how to set up synthetic data generation workflows that can produce diverse, perfectly-labeled training data for computer vision tasks, which is essential for developing robust perception systems for robotics applications.