---
title: "Voice-to-Text: Setting Up Speech Recognition with OpenAI Whisper"
sidebar_position: 2
---

# Voice-to-Text: Setting Up Speech Recognition with OpenAI Whisper

## Overview

This chapter focuses on implementing speech recognition capabilities using OpenAI Whisper, a state-of-the-art automatic speech recognition (ASR) system. We'll explore how to set up Whisper for real-time voice command processing, integrate it with our Isaac Sim environment, and optimize it for robotics applications.

## Understanding OpenAI Whisper

OpenAI Whisper is a robust speech recognition model that can handle various accents, languages, and acoustic conditions. For robotics applications, Whisper provides:

- **Multilingual Support**: Works with multiple languages for international robotics applications
- **Robustness**: Handles background noise and varying audio quality
- **Accuracy**: High transcription accuracy for clear commands
- **Flexibility**: Can run locally or via API depending on computational requirements

### Whisper Model Variants

Whisper comes in different sizes with trade-offs between accuracy, speed, and resource requirements:

| Model | Size | Required VRAM | Relative Speed | English-only | Multilingual |
|-------|------|---------------|----------------|--------------|--------------|
| tiny  | 75MB | ~1GB | ~32x | ✓ | ✓ |
| base  | 145MB | ~1GB | ~16x | ✓ | ✓ |
| small | 485MB | ~2GB | ~6x | ✓ | ✓ |
| medium | 1.5GB | ~5GB | ~2x | ✓ | ✓ |
| large | 3.0GB | ~10GB | 1x | ✗ | ✓ |

For robotics applications, we recommend the `small` or `medium` models for a good balance of performance and accuracy.

## Setting Up Whisper for Robotics

### Installation and Dependencies

```bash
# Install Whisper and related dependencies
pip3 install openai-whisper
pip3 install sounddevice
pip3 install pyaudio
pip3 install numpy
pip3 install scipy

# For faster processing, install faster-whisper (alternative implementation)
pip3 install faster-whisper
```

### Basic Whisper Implementation

```python
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time

class WhisperRobotInterface:
    def __init__(self, model_size="small", device="cuda"):
        """
        Initialize Whisper interface for robot voice commands
        """
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=device)
        print("Model loaded successfully")

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio input
        """
        if status:
            print(status)
        # Add audio chunk to queue
        self.audio_queue.put(indata.copy())

    def start_listening(self):
        """
        Start listening for voice commands
        """
        self.is_listening = True

        # Start audio stream
        with sd.InputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=self.sample_rate
        ):
            print("Listening for voice commands...")

            while self.is_listening:
                try:
                    # Get audio chunk from queue
                    audio_chunk = self.audio_queue.get(timeout=1.0)

                    # Convert to float32 and flatten
                    audio_data = audio_chunk.flatten().astype(np.float32)

                    # Process if audio level is above threshold
                    if np.max(np.abs(audio_data)) > 0.01:  # Simple voice activity detection
                        self.process_audio_chunk(audio_data)

                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break

    def process_audio_chunk(self, audio_data):
        """
        Process audio chunk with Whisper
        """
        # Transcribe the audio
        result = self.model.transcribe(
            audio_data,
            language="en",
            task="transcribe",
            fp16=True  # Use float16 for faster inference
        )

        # Check if transcription is meaningful
        transcription = result["text"].strip()
        if len(transcription) > 3:  # Ignore very short transcriptions
            confidence = result.get("avg_logprob", -1.0)  # Log probability as confidence measure
            if confidence > -1.0:  # Threshold for acceptable confidence
                self.handle_voice_command(transcription, confidence)

    def handle_voice_command(self, command, confidence):
        """
        Handle the recognized voice command
        """
        print(f"Recognized command: '{command}' (confidence: {confidence:.2f})")

        # In a real robot system, this would trigger further processing
        # For now, we'll just print it
        self.process_robot_command(command)

    def process_robot_command(self, command):
        """
        Process the voice command for robot execution
        """
        # This would typically send the command to the LLM for interpretation
        # For now, we'll just log it
        print(f"Processing robot command: {command}")

    def stop_listening(self):
        """
        Stop the listening process
        """
        self.is_listening = False

# Example usage
if __name__ == "__main__":
    whisper_interface = WhisperRobotInterface(model_size="small")

    try:
        whisper_interface.start_listening()
    except KeyboardInterrupt:
        print("\nStopping voice recognition...")
        whisper_interface.stop_listening()
```

## Optimizing Whisper for Real-Time Robotics

### Performance Optimization Techniques

For robotics applications, real-time performance is crucial. Here are key optimization strategies:

```python
import torch
from faster_whisper import WhisperModel
import asyncio

class OptimizedWhisperInterface:
    def __init__(self, model_size="small", device="cuda"):
        """
        Optimized Whisper interface using faster-whisper
        """
        self.model_size = model_size
        self.device = device

        # Load model with optimizations
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )

        # Cache for repeated processing
        self.transcription_cache = {}
        self.cache_size_limit = 100

    def transcribe_audio_optimized(self, audio_data, language="en"):
        """
        Optimized transcription with caching and batching
        """
        # Create a hash of the audio data for caching
        audio_hash = hash(tuple(audio_data[:100]))  # Hash first 100 samples

        if audio_hash in self.transcription_cache:
            return self.transcription_cache[audio_hash]

        # Perform transcription
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            best_of=5,
            patience=1.0
        )

        # Extract text from segments
        text = "".join([segment.text for segment in segments])

        # Add to cache with size limiting
        if len(self.transcription_cache) < self.cache_size_limit:
            self.transcription_cache[audio_hash] = {
                "text": text,
                "language": info.language,
                "confidence": info.avg_logprob
            }

        return {
            "text": text,
            "language": info.language,
            "confidence": info.avg_logprob
        }
```

## Integration with Isaac Sim

### Voice Command Processing Pipeline

Now let's create a ROS 2 node that integrates Whisper with Isaac Sim:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import numpy as np
import threading
import queue

class IsaacWhisperNode(Node):
    def __init__(self):
        super().__init__('isaac_whisper_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("small", device="cuda")

        # Publisher for transcribed commands
        self.command_pub = self.create_publisher(String, '/voice_commands', 10)

        # Publisher for robot responses
        self.response_pub = self.create_publisher(String, '/robot_responses', 10)

        # Audio buffer for processing
        self.audio_buffer = []
        self.buffer_size = 16000 * 2  # 2 seconds of audio at 16kHz

        # Timer for processing audio periodically
        self.process_timer = self.create_timer(2.0, self.process_audio_buffer)

        # Initialize audio input (simplified - in practice, use PyAudio or similar)
        self.audio_queue = queue.Queue()

        self.get_logger().info("Isaac Whisper Node initialized")

    def audio_callback(self, audio_msg):
        """
        Callback for audio data from Isaac Sim
        """
        # Convert audio data to numpy array
        audio_data = np.frombuffer(audio_msg.data, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0  # Normalize to [-1, 1]

        # Add to buffer
        self.audio_buffer.extend(audio_data)

        # Trim buffer if too large
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]

    def process_audio_buffer(self):
        """
        Process accumulated audio buffer with Whisper
        """
        if len(self.audio_buffer) < 16000:  # At least 1 second of audio
            return

        # Convert buffer to numpy array
        audio_array = np.array(self.audio_buffer)

        # Check if there's significant audio energy (simple VAD)
        if np.max(np.abs(audio_array)) < 0.02:  # Below threshold
            return

        try:
            # Perform transcription
            result = self.whisper_model.transcribe(
                audio_array,
                language="en",
                task="transcribe",
                fp16=True
            )

            transcription = result["text"].strip()
            confidence = result.get("avg_logprob", -2.0)

            if len(transcription) > 3 and confidence > -1.0:
                # Publish the recognized command
                cmd_msg = String()
                cmd_msg.data = f"{transcription} (confidence: {confidence:.2f})"
                self.command_pub.publish(cmd_msg)

                self.get_logger().info(f"Transcribed: {transcription}")

                # Process the command further (send to LLM for interpretation)
                self.process_robot_command(transcription)

        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")

    def process_robot_command(self, command):
        """
        Process the voice command for robot execution
        """
        # In a complete system, this would send the command to the LLM
        # for interpretation and action planning
        self.get_logger().info(f"Processing robot command: {command}")

        # For now, send a simple response
        response_msg = String()
        response_msg.data = f"Understood: {command}. Processing..."
        self.response_pub.publish(response_msg)

def main(args=None):
    rclpy.init(args=args)

    node = IsaacWhisperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Isaac Whisper Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Issues

### Audio Input Problems

```python
def troubleshoot_audio_input():
    """
    Helper function to diagnose audio input issues
    """
    import pyaudio

    # List available audio devices
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} - Channels: {info['maxInputChannels']}")

    # Test basic audio recording
    try:
        import sounddevice as sd
        print("Testing audio input...")
        test_audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        print("Audio input test successful")
        print(f"Recorded {len(test_audio)} samples")
        print(f"Max amplitude: {np.max(np.abs(test_audio))}")
    except Exception as e:
        print(f"Audio test failed: {e}")
```

### Whisper Model Loading Issues

```python
def load_whisper_with_fallback():
    """
    Load Whisper model with fallback options
    """
    import whisper
    import torch

    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Try different model sizes based on available memory
    model_sizes = ["tiny", "base", "small", "medium"]

    for size in model_sizes:
        try:
            print(f"Attempting to load Whisper {size} model...")
            model = whisper.load_model(size, device=device)
            print(f"Successfully loaded Whisper {size} model")
            return model
        except RuntimeError as e:
            print(f"Failed to load Whisper {size}: {e}")
            continue

    raise Exception("Could not load any Whisper model - check system requirements")
```

## Performance Considerations

### Real-Time Processing Requirements

For robotics applications, real-time performance is critical:

- **Latency**: Voice-to-text conversion should complete within 500ms for natural interaction
- **Throughput**: System should handle continuous audio input without dropping samples
- **Memory**: Model loading and inference should fit within available GPU memory
- **Power**: Processing should not exceed thermal limits of robotic platform

### Resource Optimization

```python
def optimize_resources():
    """
    Optimize system resources for Whisper processing
    """
    import gc
    import torch

    # Clear any existing cached models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set memory fraction if using CUDA
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
```

## Summary

This chapter covered the setup and implementation of speech recognition using OpenAI Whisper for robotics applications. We explored different Whisper model variants, implemented optimized transcription pipelines, and created ROS 2 integration with Isaac Sim. The voice-to-text component forms the foundation for our complete VLA pipeline, enabling robots to understand natural language commands through speech.

This chapter connects to:
- [Chapter 1: Introduction to VLA Robotics](./01-introduction-to-vla-robotics.md) - Provides the overall context for VLA systems
- [Chapter 3: Natural Language with LLMs](./03-natural-language-with-llms.md) - Takes the transcribed text and interprets it with LLMs
- [Chapter 4: Cognitive Planning for ROS Actions](./04-cognitive-planning-ros-actions.md) - Uses the interpreted commands to generate action plans

In the next chapter, we'll dive into understanding natural language commands with Large Language Models (LLMs), where we'll interpret the transcribed text and convert it into structured robot actions.