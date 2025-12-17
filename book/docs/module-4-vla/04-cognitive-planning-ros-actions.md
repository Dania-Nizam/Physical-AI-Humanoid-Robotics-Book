---
title: "Cognitive Planning: Converting Natural Language to ROS 2 Actions"
sidebar_position: 4
---

# Cognitive Planning: Converting Natural Language to ROS 2 Actions

## Overview

Cognitive planning represents the bridge between high-level natural language commands and low-level robot actions. This chapter explores how to translate human instructions into structured ROS 2 action sequences that can be executed by robots in simulation. We'll implement planning algorithms that can interpret natural language, generate action plans, and execute them in Isaac Sim.

## Understanding Cognitive Planning in Robotics

Cognitive planning in robotics involves several key components:

- **Intent Recognition**: Understanding what the user wants the robot to accomplish
- **Action Decomposition**: Breaking high-level goals into executable steps
- **Plan Generation**: Creating sequences of actions with proper dependencies
- **Execution Monitoring**: Tracking plan execution and handling failures

### Hierarchical Task Networks (HTN) for Robotics

Hierarchical Task Networks provide a powerful framework for decomposing complex robotic tasks:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import json

class ActionType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    WAIT = "wait"

@dataclass
class ActionStep:
    """Represents a single action in a plan"""
    action_type: ActionType
    action_name: str
    parameters: Dict[str, Any]
    duration_estimate: float  # in seconds
    preconditions: List[str]
    effects: List[str]
    priority: int = 1

class ActionLibrary:
    """Library of available robot actions"""

    def __init__(self):
        self.actions = {
            # Navigation actions
            "move_to": ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                duration_estimate=5.0,
                preconditions=["robot_is_operational"],
                effects=["robot_at_target_position"]
            ),
            "navigate_to_object": ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="navigate_to_object",
                parameters={"object_name": "unknown", "approach_distance": 1.0},
                duration_estimate=8.0,
                preconditions=["object_detected"],
                effects=["robot_at_object_approach"]
            ),

            # Manipulation actions
            "pick_up": ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="pick_up",
                parameters={"object_name": "unknown", "gripper_force": 50.0},
                duration_estimate=3.0,
                preconditions=["robot_at_object_approach", "object_graspable"],
                effects=["object_in_gripper"]
            ),
            "place_at": ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="place_at",
                parameters={"target_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                duration_estimate=3.0,
                preconditions=["object_in_gripper"],
                effects=["object_placed", "gripper_empty"]
            ),
            "grasp_object": ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="grasp_object",
                parameters={"object_name": "unknown", "gripper_position": 0.0},
                duration_estimate=2.0,
                preconditions=["robot_at_object_approach", "object_reachable"],
                effects=["object_grasped"]
            ),

            # Perception actions
            "detect_object": ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="detect_object",
                parameters={"object_type": "unknown", "search_area": "current_view"},
                duration_estimate=2.0,
                preconditions=["camera_operational"],
                effects=["object_detected"]
            ),
            "scan_environment": ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="scan_environment",
                parameters={"scan_radius": 2.0},
                duration_estimate=5.0,
                preconditions=["lidar_operational"],
                effects=["environment_mapped"]
            ),

            # Communication actions
            "speak_response": ActionStep(
                action_type=ActionType.COMMUNICATION,
                action_name="speak_response",
                parameters={"message": "unknown"},
                duration_estimate=1.0,
                preconditions=["tts_system_operational"],
                effects=["message_delivered"]
            ),

            # Wait actions
            "wait_for_completion": ActionStep(
                action_type=ActionType.WAIT,
                action_name="wait_for_completion",
                parameters={"timeout": 10.0},
                duration_estimate=0.0,
                preconditions=[],
                effects=["task_completed"]
            )
        }

    def get_action(self, action_name: str) -> Optional[ActionStep]:
        """Get an action from the library"""
        return self.actions.get(action_name)

    def add_action(self, action: ActionStep):
        """Add a new action to the library"""
        self.actions[action.action_name] = action

class HTNPlanner:
    """Hierarchical Task Network Planner for Robotics"""

    def __init__(self):
        self.action_library = ActionLibrary()
        self.task_decomposition_rules = {
            "fetch_object": self._decompose_fetch_object,
            "move_object": self._decompose_move_object,
            "inspect_area": self._decompose_inspect_area,
            "navigate_and_report": self._decompose_navigate_and_report
        }

    def _decompose_fetch_object(self, parameters: Dict[str, Any]) -> List[ActionStep]:
        """Decompose fetch object task into primitive actions"""
        object_name = parameters.get("object_name", "unknown")
        target_location = parameters.get("target_location", {"x": 0.0, "y": 0.0, "z": 0.0})

        return [
            ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="detect_object",
                parameters={"object_type": object_name},
                duration_estimate=2.0,
                preconditions=["camera_operational"],
                effects=["object_detected"]
            ),
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="navigate_to_object",
                parameters={"object_name": object_name, "approach_distance": 1.0},
                duration_estimate=8.0,
                preconditions=["object_detected"],
                effects=["robot_at_object_approach"]
            ),
            ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="pick_up",
                parameters={"object_name": object_name, "gripper_force": 50.0},
                duration_estimate=3.0,
                preconditions=["robot_at_object_approach", "object_graspable"],
                effects=["object_in_gripper"]
            ),
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": target_location},
                duration_estimate=5.0,
                preconditions=["object_in_gripper"],
                effects=["robot_at_target_position"]
            ),
            ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="place_at",
                parameters={"target_position": target_location},
                duration_estimate=3.0,
                preconditions=["object_in_gripper", "robot_at_target_position"],
                effects=["object_placed", "gripper_empty"]
            )
        ]

    def _decompose_move_object(self, parameters: Dict[str, Any]) -> List[ActionStep]:
        """Decompose move object task into primitive actions"""
        object_name = parameters.get("object_name", "unknown")
        start_location = parameters.get("start_location", {"x": 0.0, "y": 0.0, "z": 0.0})
        end_location = parameters.get("end_location", {"x": 0.0, "y": 0.0, "z": 0.0})

        return [
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": start_location},
                duration_estimate=5.0,
                preconditions=["robot_is_operational"],
                effects=["robot_at_start_location"]
            ),
            ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="detect_object",
                parameters={"object_type": object_name},
                duration_estimate=2.0,
                preconditions=["robot_at_start_location", "camera_operational"],
                effects=["object_detected"]
            ),
            ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="pick_up",
                parameters={"object_name": object_name, "gripper_force": 50.0},
                duration_estimate=3.0,
                preconditions=["object_detected", "object_graspable"],
                effects=["object_in_gripper"]
            ),
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": end_location},
                duration_estimate=5.0,
                preconditions=["object_in_gripper"],
                effects=["robot_at_end_location"]
            ),
            ActionStep(
                action_type=ActionType.MANIPULATION,
                action_name="place_at",
                parameters={"target_position": end_location},
                duration_estimate=3.0,
                preconditions=["object_in_gripper", "robot_at_end_location"],
                effects=["object_placed", "gripper_empty"]
            )
        ]

    def _decompose_inspect_area(self, parameters: Dict[str, Any]) -> List[ActionStep]:
        """Decompose inspect area task into primitive actions"""
        area_center = parameters.get("area_center", {"x": 0.0, "y": 0.0, "z": 0.0})
        area_radius = parameters.get("area_radius", 2.0)

        return [
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": area_center},
                duration_estimate=5.0,
                preconditions=["robot_is_operational"],
                effects=["robot_at_area_center"]
            ),
            ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="scan_environment",
                parameters={"scan_radius": area_radius},
                duration_estimate=5.0,
                preconditions=["robot_at_area_center", "lidar_operational"],
                effects=["environment_mapped"]
            ),
            ActionStep(
                action_type=ActionType.PERCEPTION,
                action_name="detect_object",
                parameters={"object_type": "any", "search_area": "mapped_area"},
                duration_estimate=3.0,
                preconditions=["environment_mapped"],
                effects=["objects_in_area_detected"]
            )
        ]

    def _decompose_navigate_and_report(self, parameters: Dict[str, Any]) -> List[ActionStep]:
        """Decompose navigate and report task into primitive actions"""
        target_position = parameters.get("target_position", {"x": 0.0, "y": 0.0, "z": 0.0})
        report_message = parameters.get("report_message", "Arrived at destination")

        return [
            ActionStep(
                action_type=ActionType.NAVIGATION,
                action_name="move_to",
                parameters={"target_position": target_position},
                duration_estimate=5.0,
                preconditions=["robot_is_operational"],
                effects=["robot_at_target_position"]
            ),
            ActionStep(
                action_type=ActionType.COMMUNICATION,
                action_name="speak_response",
                parameters={"message": report_message},
                duration_estimate=1.0,
                preconditions=["robot_at_target_position", "tts_system_operational"],
                effects=["message_delivered"]
            )
        ]

    def plan_task(self, task_name: str, parameters: Dict[str, Any]) -> Optional[List[ActionStep]]:
        """Generate a plan for a high-level task"""
        if task_name in self.task_decomposition_rules:
            return self.task_decomposition_rules[task_name](parameters)
        else:
            print(f"Unknown task: {task_name}")
            return None

# Example usage
if __name__ == "__main__":
    planner = HTNPlanner()

    # Example: Fetch a red cube and place it at a specific location
    fetch_plan = planner.plan_task("fetch_object", {
        "object_name": "red_cube",
        "target_location": {"x": 1.0, "y": 2.0, "z": 0.5}
    })

    if fetch_plan:
        print("Generated plan for fetching object:")
        for i, action in enumerate(fetch_plan):
            print(f"{i+1}. {action.action_name} - {action.parameters}")
    else:
        print("Failed to generate plan")
```

## PDDL-Style Planning for Complex Tasks

While HTN planning works well for structured tasks, PDDL-style planning provides more flexibility for complex reasoning:

```python
class PDDLStylePlanner:
    """PDDL-style planner for complex robotic tasks"""

    def __init__(self):
        self.state = {
            "robot_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "gripper_state": "empty",  # "empty" or "holding_object_name"
            "objects": {},  # object_name -> position
            "robot_operational": True,
            "camera_operational": True,
            "lidar_operational": True,
            "tts_operational": True
        }
        self.action_library = ActionLibrary()

    def update_state(self, action: ActionStep):
        """Update the world state based on action execution"""
        if action.action_name == "move_to":
            target_pos = action.parameters["target_position"]
            self.state["robot_position"] = target_pos
        elif action.action_name == "pick_up":
            object_name = action.parameters["object_name"]
            self.state["gripper_state"] = f"holding_{object_name}"
        elif action.action_name == "place_at":
            if self.state["gripper_state"].startswith("holding_"):
                held_object = self.state["gripper_state"].replace("holding_", "")
                self.state["gripper_state"] = "empty"
                # Update object position to where it was placed
                self.state["objects"][held_object] = action.parameters["target_position"]

    def check_preconditions(self, action: ActionStep) -> bool:
        """Check if action preconditions are satisfied"""
        for precondition in action.preconditions:
            if precondition == "robot_is_operational":
                if not self.state["robot_operational"]:
                    return False
            elif precondition == "camera_operational":
                if not self.state["camera_operational"]:
                    return False
            elif precondition == "lidar_operational":
                if not self.state["lidar_operational"]:
                    return False
            elif precondition == "tts_system_operational":
                if not self.state["tts_operational"]:
                    return False
            elif precondition == "object_in_gripper":
                if self.state["gripper_state"] == "empty":
                    return False
            elif precondition == "gripper_empty":
                if self.state["gripper_state"] != "empty":
                    return False
            # Add more precondition checks as needed

        return True

    def plan_with_pddl_style(self, goal_condition: str, goal_parameters: Dict[str, Any]) -> Optional[List[ActionStep]]:
        """Generate plan using PDDL-style reasoning"""
        # This is a simplified version - in practice, this would use a proper PDDL solver
        if goal_condition == "object_at_location":
            object_name = goal_parameters["object_name"]
            target_location = goal_parameters["target_location"]

            # Check if we know where the object is
            if object_name not in self.state["objects"]:
                # Need to find the object first
                return [
                    ActionStep(
                        action_type=ActionType.PERCEPTION,
                        action_name="detect_object",
                        parameters={"object_type": object_name},
                        duration_estimate=2.0,
                        preconditions=["camera_operational"],
                        effects=["object_detected"]
                    ),
                    ActionStep(
                        action_type=ActionType.NAVIGATION,
                        action_name="navigate_to_object",
                        parameters={"object_name": object_name, "approach_distance": 1.0},
                        duration_estimate=8.0,
                        preconditions=["object_detected"],
                        effects=["robot_at_object_approach"]
                    ),
                    ActionStep(
                        action_type=ActionType.MANIPULATION,
                        action_name="pick_up",
                        parameters={"object_name": object_name, "gripper_force": 50.0},
                        duration_estimate=3.0,
                        preconditions=["robot_at_object_approach", "object_graspable"],
                        effects=["object_in_gripper"]
                    ),
                    ActionStep(
                        action_type=ActionType.NAVIGATION,
                        action_name="move_to",
                        parameters={"target_position": target_location},
                        duration_estimate=5.0,
                        preconditions=["object_in_gripper"],
                        effects=["robot_at_target_position"]
                    ),
                    ActionStep(
                        action_type=ActionType.MANIPULATION,
                        action_name="place_at",
                        parameters={"target_position": target_location},
                        duration_estimate=3.0,
                        preconditions=["object_in_gripper", "robot_at_target_position"],
                        effects=["object_placed", "gripper_empty"]
                    )
                ]
            else:
                # Object location is known, plan directly
                current_obj_pos = self.state["objects"][object_name]
                return [
                    ActionStep(
                        action_type=ActionType.NAVIGATION,
                        action_name="move_to",
                        parameters={"target_position": current_obj_pos},
                        duration_estimate=5.0,
                        preconditions=["robot_is_operational"],
                        effects=["robot_at_object_location"]
                    ),
                    ActionStep(
                        action_type=ActionType.MANIPULATION,
                        action_name="pick_up",
                        parameters={"object_name": object_name, "gripper_force": 50.0},
                        duration_estimate=3.0,
                        preconditions=["robot_at_object_location", "object_graspable"],
                        effects=["object_in_gripper"]
                    ),
                    ActionStep(
                        action_type=ActionType.NAVIGATION,
                        action_name="move_to",
                        parameters={"target_position": target_location},
                        duration_estimate=5.0,
                        preconditions=["object_in_gripper"],
                        effects=["robot_at_target_position"]
                    ),
                    ActionStep(
                        action_type=ActionType.MANIPULATION,
                        action_name="place_at",
                        parameters={"target_position": target_location},
                        duration_estimate=3.0,
                        preconditions=["object_in_gripper", "robot_at_target_position"],
                        effects=["object_placed", "gripper_empty"]
                    )
                ]

        return None
```

## Language-to-Action Translation

The key challenge in VLA systems is translating natural language commands into structured action plans:

```python
import re
from typing import Tuple

class LanguageToActionTranslator:
    """Translates natural language commands to action plans"""

    def __init__(self):
        self.planner = HTNPlanner()
        self.patterns = {
            # Fetch object patterns
            "fetch": [
                r"pick up the (.+)",
                r"grab the (.+)",
                r"get the (.+)",
                r"fetch the (.+)",
                r"bring me the (.+)"
            ],
            # Navigation patterns
            "navigate": [
                r"go to (.+)",
                r"move to (.+)",
                r"navigate to (.+)",
                r"walk to (.+)"
            ],
            # Move object patterns
            "move_object": [
                r"move the (.+) to (.+)",
                r"put the (.+) at (.+)",
                r"place the (.+) on (.+)"
            ],
            # Inspect patterns
            "inspect": [
                r"look at (.+)",
                r"inspect (.+)",
                r"scan (.+)",
                r"detect (.+)"
            ]
        }

    def parse_command(self, command: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a natural language command into task and parameters"""
        command_lower = command.lower().strip()

        # Try fetch patterns
        for pattern in self.patterns["fetch"]:
            match = re.search(pattern, command_lower)
            if match:
                return "fetch_object", {"object_name": match.group(1).strip()}

        # Try navigate patterns
        for pattern in self.patterns["navigate"]:
            match = re.search(pattern, command_lower)
            if match:
                location = match.group(1).strip()
                # Convert location descriptions to coordinates (simplified)
                coordinates = self._convert_location_to_coordinates(location)
                return "navigate_and_report", {
                    "target_position": coordinates,
                    "report_message": f"Arrived at {location}"
                }

        # Try move object patterns
        for pattern in self.patterns["move_object"]:
            match = re.search(pattern, command_lower)
            if match:
                object_name = match.group(1).strip()
                target_location = match.group(2).strip()
                target_coords = self._convert_location_to_coordinates(target_location)

                # Find object location (simplified - assume it's at origin for now)
                start_coords = {"x": 0.0, "y": 0.0, "z": 0.0}

                return "move_object", {
                    "object_name": object_name,
                    "start_location": start_coords,
                    "end_location": target_coords
                }

        # Try inspect patterns
        for pattern in self.patterns["inspect"]:
            match = re.search(pattern, command_lower)
            if match:
                location = match.group(1).strip()
                coordinates = self._convert_location_to_coordinates(location)
                return "inspect_area", {
                    "area_center": coordinates,
                    "area_radius": 2.0
                }

        # If no pattern matches, return a generic task
        return "speak_response", {
            "message": f"I don't understand the command: {command}"
        }

    def _convert_location_to_coordinates(self, location_desc: str) -> Dict[str, float]:
        """Convert location descriptions to coordinates (simplified mapping)"""
        location_map = {
            "kitchen": {"x": 2.0, "y": 1.0, "z": 0.0},
            "living room": {"x": -1.0, "y": 0.0, "z": 0.0},
            "bedroom": {"x": 0.0, "y": -2.0, "z": 0.0},
            "office": {"x": 1.5, "y": -1.0, "z": 0.0},
            "table": {"x": 0.5, "y": 0.5, "z": 0.0},
            "couch": {"x": -0.5, "y": 1.0, "z": 0.0}
        }

        # Check if location is in our map
        if location_desc in location_map:
            return location_map[location_desc]

        # If not found, return a default position
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    def translate_command(self, command: str) -> Optional[List[ActionStep]]:
        """Translate a natural language command to an action plan"""
        task_name, parameters = self.parse_command(command)
        return self.planner.plan_task(task_name, parameters)

# Example usage
if __name__ == "__main__":
    translator = LanguageToActionTranslator()

    # Test various commands
    commands = [
        "Pick up the red cube",
        "Go to the kitchen",
        "Move the blue ball to the table",
        "Look at the living room"
    ]

    for cmd in commands:
        print(f"\nCommand: {cmd}")
        plan = translator.translate_command(cmd)
        if plan:
            print("Generated plan:")
            for i, action in enumerate(plan):
                print(f"  {i+1}. {action.action_name} - {action.parameters}")
        else:
            print("Could not generate plan for this command")
```

## ROS 2 Integration for Action Execution

Now let's create a ROS 2 node that integrates our cognitive planning with Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus

# Isaac ROS action imports (these would need to be defined based on your Isaac packages)
# from isaac_ros_messages.action import NavigateToPose, ManipulateObject

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner_node')

        # Initialize planners
        self.htn_planner = HTNPlanner()
        self.pddl_planner = PDDLStylePlanner()
        self.language_translator = LanguageToActionTranslator()

        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_commands',
            self.command_callback,
            10
        )

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            String,
            '/action_sequence',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            '/planning_feedback',
            10
        )

        # Action clients for Isaac Sim integration
        # self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # self.manip_client = ActionClient(self, ManipulateObject, 'manipulate_object')

        # Current plan and execution state
        self.current_plan = []
        self.current_plan_index = 0
        self.is_executing = False

        self.get_logger().info("Cognitive Planner Node initialized")

    def command_callback(self, msg):
        """Handle incoming natural language commands"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        # Translate command to action plan
        plan = self.language_translator.translate_command(command)

        if plan:
            self.get_logger().info(f"Generated plan with {len(plan)} actions")

            # Publish the action sequence
            plan_json = json.dumps([
                {
                    "action_name": action.action_name,
                    "parameters": action.parameters,
                    "action_type": action.action_type.value
                }
                for action in plan
            ])

            action_msg = String()
            action_msg.data = plan_json
            self.action_sequence_pub.publish(action_msg)

            # Start execution
            self.execute_plan(plan)
        else:
            feedback_msg = String()
            feedback_msg.data = f"Could not understand command: {command}"
            self.feedback_pub.publish(feedback_msg)

    def execute_plan(self, plan: List[ActionStep]):
        """Execute the generated plan step by step"""
        if not plan:
            self.get_logger().warn("Cannot execute empty plan")
            return

        self.current_plan = plan
        self.current_plan_index = 0
        self.is_executing = True

        self.get_logger().info(f"Starting execution of plan with {len(plan)} actions")

        # Execute first action
        self.execute_next_action()

    def execute_next_action(self):
        """Execute the next action in the plan"""
        if not self.is_executing or self.current_plan_index >= len(self.current_plan):
            self.is_executing = False
            self.get_logger().info("Plan execution completed")

            # Send completion feedback
            feedback_msg = String()
            feedback_msg.data = "Plan execution completed successfully"
            self.feedback_pub.publish(feedback_msg)
            return

        current_action = self.current_plan[self.current_plan_index]
        self.get_logger().info(f"Executing action {self.current_plan_index + 1}/{len(self.current_plan)}: {current_action.action_name}")

        # Check preconditions
        if not self.pddl_planner.check_preconditions(current_action):
            self.get_logger().error(f"Preconditions not met for action: {current_action.action_name}")
            self.is_executing = False

            feedback_msg = String()
            feedback_msg.data = f"Preconditions not met for action: {current_action.action_name}"
            self.feedback_pub.publish(feedback_msg)
            return

        # Execute the action based on its type
        if current_action.action_type == ActionType.NAVIGATION:
            self.execute_navigation_action(current_action)
        elif current_action.action_type == ActionType.MANIPULATION:
            self.execute_manipulation_action(current_action)
        elif current_action.action_type == ActionType.PERCEPTION:
            self.execute_perception_action(current_action)
        elif current_action.action_type == ActionType.COMMUNICATION:
            self.execute_communication_action(current_action)
        elif current_action.action_type == ActionType.WAIT:
            self.execute_wait_action(current_action)

    def execute_navigation_action(self, action: ActionStep):
        """Execute navigation action"""
        # This would typically call an Isaac Sim navigation service
        target_pos = action.parameters["target_position"]

        self.get_logger().info(f"Navigating to position: {target_pos}")

        # Simulate navigation completion after a delay
        self.create_timer(action.duration_estimate, self.on_action_completed)

    def execute_manipulation_action(self, action: ActionStep):
        """Execute manipulation action"""
        # This would typically call an Isaac Sim manipulation service
        self.get_logger().info(f"Executing manipulation: {action.action_name}")

        # Simulate manipulation completion after a delay
        self.create_timer(action.duration_estimate, self.on_action_completed)

    def execute_perception_action(self, action: ActionStep):
        """Execute perception action"""
        # This would typically call an Isaac Sim perception service
        self.get_logger().info(f"Executing perception: {action.action_name}")

        # Simulate perception completion after a delay
        self.create_timer(action.duration_estimate, self.on_action_completed)

    def execute_communication_action(self, action: ActionStep):
        """Execute communication action"""
        message = action.parameters["message"]
        self.get_logger().info(f"Communicating: {message}")

        # Simulate communication completion
        self.create_timer(action.duration_estimate, self.on_action_completed)

    def execute_wait_action(self, action: ActionStep):
        """Execute wait action"""
        timeout = action.parameters["timeout"]
        self.get_logger().info(f"Waiting for {timeout} seconds")

        # Wait for specified time
        self.create_timer(timeout, self.on_action_completed)

    def on_action_completed(self):
        """Callback when an action completes"""
        # Update the world state
        completed_action = self.current_plan[self.current_plan_index]
        self.pddl_planner.update_state(completed_action)

        self.get_logger().info(f"Action completed: {completed_action.action_name}")

        # Move to next action
        self.current_plan_index += 1
        self.execute_next_action()

def main(args=None):
    rclpy.init(args=args)

    node = CognitivePlannerNode()

    try:
        # Use multi-threaded executor to handle callbacks properly
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        node.get_logger().info("Starting cognitive planner node...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cognitive planner node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Isaac Sim

To fully integrate with Isaac Sim, we need to create the necessary action definitions and message types:

```python
# IsaacSimActionBridge.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

class IsaacSimActionBridge(Node):
    """Bridge between ROS 2 action plans and Isaac Sim execution"""

    def __init__(self):
        super().__init__('isaac_sim_action_bridge')

        # Subscribers for action sequences
        self.action_sub = self.create_subscription(
            String,
            '/action_sequence',
            self.action_sequence_callback,
            10
        )

        # Publishers for Isaac Sim commands
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/isaac_sim/joint_commands',
            10
        )

        self.nav_goal_pub = self.create_publisher(
            Pose,
            '/isaac_sim/navigation_goal',
            10
        )

        # Subscriber for Isaac Sim state
        self.joint_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/isaac_sim/joint_states',
            self.joint_state_callback,
            10
        )

        # Action execution state
        self.current_action_sequence = []
        self.action_index = 0
        self.is_executing = False

        self.get_logger().info("Isaac Sim Action Bridge initialized")

    def action_sequence_callback(self, msg):
        """Receive action sequence and start execution"""
        try:
            action_sequence = json.loads(msg.data)
            self.get_logger().info(f"Received action sequence with {len(action_sequence)} actions")

            self.current_action_sequence = action_sequence
            self.action_index = 0
            self.is_executing = True

            self.execute_next_action()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse action sequence: {e}")

    def execute_next_action(self):
        """Execute the next action in the sequence"""
        if not self.is_executing or self.action_index >= len(self.current_action_sequence):
            self.is_executing = False
            self.get_logger().info("Action sequence execution completed")
            return

        action = self.current_action_sequence[self.action_index]
        self.get_logger().info(f"Executing action {self.action_index + 1}: {action['action_name']}")

        # Execute based on action type
        if action['action_type'] == 'navigation':
            self.execute_navigation(action)
        elif action['action_type'] == 'manipulation':
            self.execute_manipulation(action)
        elif action['action_type'] == 'perception':
            self.execute_perception(action)
        elif action['action_type'] == 'communication':
            self.execute_communication(action)

    def execute_navigation(self, action):
        """Execute navigation action in Isaac Sim"""
        target_pos = action['parameters']['target_position']

        # Create navigation goal message
        nav_goal = Pose()
        nav_goal.position.x = target_pos['x']
        nav_goal.position.y = target_pos['y']
        nav_goal.position.z = target_pos['z']
        nav_goal.orientation.w = 1.0  # Default orientation

        self.nav_goal_pub.publish(nav_goal)

        # Schedule completion check
        self.create_timer(action['duration_estimate'], self.on_action_completed)

    def execute_manipulation(self, action):
        """Execute manipulation action in Isaac Sim"""
        # This would send joint commands to Isaac Sim
        joint_cmd = JointState()
        joint_cmd.name = ['joint1', 'joint2', 'joint3']  # Example joint names
        joint_cmd.position = [0.0, 0.0, 0.0]  # Example positions

        self.joint_cmd_pub.publish(joint_cmd)

        # Schedule completion check
        self.create_timer(action['duration_estimate'], self.on_action_completed)

    def execute_perception(self, action):
        """Execute perception action in Isaac Sim"""
        # Perception typically doesn't require direct commands
        # but might trigger sensor updates
        self.get_logger().info(f"Executing perception action: {action['action_name']}")

        # Schedule completion check
        self.create_timer(action['duration_estimate'], self.on_action_completed)

    def execute_communication(self, action):
        """Execute communication action"""
        message = action['parameters']['message']
        self.get_logger().info(f"Communicating: {message}")

        # Schedule completion check
        self.create_timer(action['duration_estimate'], self.on_action_completed)

    def on_action_completed(self):
        """Callback when action completes"""
        self.get_logger().info(f"Action {self.action_index + 1} completed")
        self.action_index += 1
        self.execute_next_action()

    def joint_state_callback(self, msg):
        """Receive joint state updates from Isaac Sim"""
        # This could be used for monitoring and feedback
        pass

def main(args=None):
    rclpy.init(args=args)

    node = IsaacSimActionBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Isaac Sim Action Bridge")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Considerations

When implementing cognitive planning for robotics, several performance factors must be considered:

- **Planning Time**: Complex plans should be generated efficiently to maintain real-time interaction
- **Memory Usage**: Large action libraries and complex state representations can consume significant memory
- **Execution Reliability**: Plans must handle failures and unexpected situations gracefully
- **Concurrency**: Multiple planning and execution threads should be properly synchronized

## Summary

This chapter covered the implementation of cognitive planning systems that convert natural language commands into executable ROS 2 action sequences. We explored both HTN and PDDL-style planning approaches, implemented language-to-action translation, and created ROS 2 integration for Isaac Sim execution. The cognitive planning component forms the crucial link between natural language understanding and robot action execution in our VLA pipeline.

This chapter connects to:
- [Chapter 3: Natural Language with LLMs](./03-natural-language-with-llms.md) - Takes the interpreted commands as input
- [Chapter 5: Integrating Perception with VLA](./05-integrating-perception-vla.md) - Uses perception data for planning
- [Chapter 6: Path Planning from Language Goals](./06-path-planning-language-goals.md) - Incorporates navigation planning

In the next chapter, we'll explore how to integrate perception systems with our VLA pipeline, enabling robots to sense and understand their environment to support more sophisticated action planning and execution.