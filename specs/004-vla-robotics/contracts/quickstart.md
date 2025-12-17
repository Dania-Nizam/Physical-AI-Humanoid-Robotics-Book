# Quickstart Guide: Vision-Language-Action (VLA) Pipeline

## Overview

This quickstart guide will help you set up and run your first Vision-Language-Action (VLA) pipeline with voice command integration. The guide assumes you have the required software installed (ROS 2 Kilted Kaiju, NVIDIA Isaac Sim 5.0, Isaac ROS 3.2) with an appropriate NVIDIA GPU and access to OpenAI Whisper and LLM APIs.

## Prerequisites

- **Hardware Requirements**:
  - NVIDIA GPU with CUDA support (RTX 4080 or higher recommended)
  - 32GB RAM or more
  - 8+ core CPU for real-time processing
  - Microphone for voice input (for real deployment)

- **Software Requirements**:
  - Ubuntu 22.04 LTS
  - NVIDIA drivers (535 or newer)
  - CUDA 12.x installed
  - ROS 2 Kilted Kaiju installed and sourced
  - Isaac Sim 5.0 installed
  - Isaac ROS 3.2 packages installed
  - Python 3.8+ with rclpy
  - OpenAI API access (or local alternatives like Ollama)

## Setup Steps

### 1. Verify System Requirements

First, verify your NVIDIA GPU and driver setup:

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify ROS 2 installation
source /opt/ros/kilted/setup.bash
ros2 topic list
```

### 2. Install Isaac Sim and Isaac ROS

If you haven't installed Isaac Sim and Isaac ROS yet:

```bash
# Isaac Sim is typically installed via Omniverse Launcher or direct download
# Verify installation
cd ~/.local/share/ov/pkg/isaac_sim-2023.1.1/  # or your installation path
python -c "import omni; print('Isaac Sim installation verified')"

# Install Isaac ROS packages
sudo apt update
sudo apt install ros-kilted-isaac-ros-common
sudo apt install ros-kilted-isaac-ros-perception
sudo apt install ros-kilted-isaac-ros-navigation
```

### 3. Install VLA Dependencies

Install the required packages for the VLA pipeline:

```bash
# Install Python dependencies
pip3 install openai
pip3 install sounddevice
pip3 install pyaudio
pip3 install numpy
pip3 install opencv-python
pip3 install whisper-openai  # or faster-whisper for local processing
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure API Access

Set up your OpenAI API access:

```bash
# Create environment file
cat > ~/.env << EOF
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_MODEL=whisper-1  # or use local model like "small" for faster-whisper
EOF

# Load the environment
source ~/.env
```

### 5. Create VLA Workspace

Create a workspace for the VLA pipeline:

```bash
# Create workspace
mkdir -p ~/vla_ws/src
cd ~/vla_ws

# Create the VLA package
ros2 pkg create --build-type ament_python vla_pipeline
cd src/vla_pipeline

# The package structure will be created during implementation
```

### 6. Basic VLA Pipeline Test

#### 6.1. Test Speech Recognition

Create a simple speech recognition test:

```python
# test_speech_recognition.py
import openai
import sounddevice as sd
import numpy as np
import wave
import io
from scipy.io.wavfile import write

def record_audio(duration=5, sample_rate=16000):
    """Record audio for speech recognition testing"""
    print(f"Recording {duration} seconds of audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for recording to complete
    print("Recording complete.")

    # Normalize audio data
    audio_data = np.squeeze(audio_data)
    audio_data = (audio_data * 32767).astype(np.int16)

    # Save to BytesIO for OpenAI API
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, audio_data)
    wav_io.seek(0)

    return wav_io.getvalue(), sample_rate

def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Save audio to temporary file for API
        with open('/tmp/temp_audio.wav', 'wb') as f:
            f.write(audio_bytes)

        # Transcribe using OpenAI API
        with open('/tmp/temp_audio.wav', 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return transcript.text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def main():
    # Set your OpenAI API key
    openai.api_key = "YOUR_API_KEY_HERE"  # Replace with actual key or use environment variable

    # Record and transcribe
    audio_bytes, sample_rate = record_audio(duration=5)
    transcription = transcribe_audio(audio_bytes)

    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("Transcription failed")

if __name__ == "__main__":
    main()
```

#### 6.2. Test LLM Integration

Create a simple LLM integration test:

```python
# test_llm_integration.py
import openai
import json

def interpret_command(transcription):
    """Interpret natural language command using LLM"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # or gpt-4-turbo
            messages=[
                {
                    "role": "system",
                    "content": """You are a robot command interpreter. Convert natural language commands to structured action sequences.
                    Respond in JSON format with: {
                        "intent": "action_type",
                        "parameters": {"key": "value"},
                        "action_sequence": [{"action": "action_type", "params": {"param": "value"}}]
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Interpret this command: '{transcription}'"
                }
            ],
            temperature=0.1
        )

        # Extract JSON from response
        content = response.choices[0].message.content
        # Find JSON in response (might be wrapped in markdown)
        if "```json" in content:
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        return json.loads(content)
    except Exception as e:
        print(f"Error interpreting command: {e}")
        return None

def main():
    # Set your OpenAI API key
    openai.api_key = "YOUR_API_KEY_HERE"  # Replace with actual key or use environment variable

    # Test command
    test_command = "Move to the red cube and pick it up"
    interpretation = interpret_command(test_command)

    if interpretation:
        print(f"Interpretation: {json.dumps(interpretation, indent=2)}")
    else:
        print("Interpretation failed")

if __name__ == "__main__":
    main()
```

### 7. Create a Simple VLA Pipeline Node

Create a basic VLA pipeline node that connects all components:

```python
# vla_pipeline_node.py
#!/usr/bin/env python3
"""
Vision-Language-Action Pipeline Node
Connects speech recognition, LLM interpretation, and robot control
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import openai
import sounddevice as sd
import numpy as np
import json
import time

class VLAPipelineNode(Node):
    def __init__(self):
        super().__init__('vla_pipeline_node')

        # Publishers for robot commands
        self.voice_cmd_pub = self.create_publisher(String, '/voice_command', 10)
        self.action_seq_pub = self.create_publisher(String, '/action_sequence', 10)

        # Initialize OpenAI
        openai.api_key = "YOUR_API_KEY_HERE"  # Should be loaded from environment

        # Audio recording parameters
        self.sample_rate = 16000
        self.record_duration = 5  # seconds
        self.is_listening = False

        # Create a timer for continuous listening
        self.listen_timer = self.create_timer(10.0, self.listen_for_commands)

        self.get_logger().info("VLA Pipeline Node initialized")

    def listen_for_commands(self):
        """Listen for voice commands and process them"""
        if self.is_listening:
            self.get_logger().info("Listening for voice command...")

            # Record audio
            audio_data = self.record_audio()

            if audio_data:
                # Transcribe audio
                transcription = self.transcribe_audio(audio_data)

                if transcription:
                    self.get_logger().info(f"Heard: {transcription}")

                    # Interpret command with LLM
                    interpretation = self.interpret_command(transcription)

                    if interpretation:
                        # Publish action sequence
                        action_msg = String()
                        action_msg.data = json.dumps(interpretation)
                        self.action_seq_pub.publish(action_msg)

                        self.get_logger().info(f"Published action sequence: {interpretation['intent']}")

    def record_audio(self, duration=None):
        """Record audio for speech recognition"""
        if duration is None:
            duration = self.record_duration

        try:
            self.get_logger().info(f"Recording {duration} seconds of audio...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete

            # Normalize and convert for API
            audio_data = np.squeeze(audio_data)
            audio_data = (audio_data * 32767).astype(np.int16)

            return audio_data
        except Exception as e:
            self.get_logger().error(f"Audio recording error: {e}")
            return None

    def transcribe_audio(self, audio_data):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Save to temporary file for API
            temp_filename = '/tmp/vla_audio.wav'
            from scipy.io.wavfile import write
            write(temp_filename, self.sample_rate, audio_data)

            # Transcribe using OpenAI API
            with open(temp_filename, 'rb') as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return transcript.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return None

    def interpret_command(self, transcription):
        """Interpret natural language command using LLM"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a robot command interpreter. Convert natural language commands to structured action sequences.
                        Respond in JSON format with: {
                            "intent": "action_type",
                            "parameters": {"key": "value"},
                            "action_sequence": [{"action": "action_type", "params": {"param": "value"}}]
                        }. Focus on navigation, manipulation, and perception tasks."""
                    },
                    {
                        "role": "user",
                        "content": f"Interpret this command: '{transcription}'. Provide only the JSON response without any additional text."
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            import re
            content = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                self.get_logger().error("Could not extract JSON from LLM response")
                return None
        except Exception as e:
            self.get_logger().error(f"Command interpretation error: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = VLAPipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VLA Pipeline Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 8. Testing the Integration

#### 8.1. Basic Voice Command Test

Run the basic test to verify speech recognition and LLM integration:

```bash
# Make sure your environment is sourced
source /opt/ros/kilted/setup.bash

# Run the voice command test
python3 test_speech_recognition.py

# Run the LLM integration test
python3 test_llm_integration.py
```

#### 8.2. Full Pipeline Test

Test the complete VLA pipeline:

```bash
# Source ROS environment
source /opt/ros/kilted/setup.bash

# Run the VLA pipeline node
python3 vla_pipeline_node.py
```

### 9. Troubleshooting Common Issues

#### Audio Input Issues
- **Problem**: No audio input detected
- **Solution**: Check microphone permissions and test with `arecord -D hw:0,0 -f cd test.wav`

#### API Connection Issues
- **Problem**: OpenAI API connection fails
- **Solution**: Verify API key and internet connection
- **Alternative**: Use local Whisper model with faster-whisper

#### Isaac Sim Connection Issues
- **Problem**: ROS 2 bridge not connecting
- **Solution**: Verify Isaac Sim is running with ROS bridge extension enabled
- **Check**: `source /opt/ros/kilted/setup.bash` and `ros2 topic list`

#### Performance Issues
- **Problem**: Slow response times
- **Solution**:
  - Use smaller LLM models for faster response
  - Reduce audio recording duration
  - Optimize Isaac Sim settings for performance

### 10. Next Steps

1. **Integrate with Isaac Sim**: Connect the VLA pipeline to a simulated humanoid robot
2. **Add perception**: Integrate Isaac ROS perception packages for object detection
3. **Enhance actions**: Implement more complex action sequences for manipulation
4. **Create capstone**: Build the complete autonomous humanoid project

This quickstart provides the foundation for implementing the Vision-Language-Action pipeline. The following chapters will dive deeper into each component and provide complete examples for voice-controlled humanoid robots.