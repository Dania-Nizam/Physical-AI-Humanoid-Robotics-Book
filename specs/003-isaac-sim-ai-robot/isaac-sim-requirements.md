# Isaac Sim 5.0 Setup Requirements

## System Requirements

### Operating System
- **Linux**: Ubuntu 22.04 (Ubuntu 24.04 is not fully supported at this time)
- **Windows**: Windows 10/11

> **Note**: Building with Ubuntu 24.04 requires GCC/G++ 11 to be installed, as GCC/G++ 12+ is not supported.

### GPU Requirements
For optimal performance, NVIDIA recommends the following GPU configurations:

#### Local Workstation
| Level | GPU |
|-------|-----|
| Minimum | RTX 4080 |
| Recommended | RTX 5080 / RTX 5880 Ada |
| Best | RTX PRO 6000 Blackwell Workstation / RTX PRO 5000 Blackwell Workstation |

#### Datacenter
| Level | GPU |
|-------|-----|
| Minimum | A40 |
| Recommended | L40S / L20 |
| Best | RTX PRO 6000 Blackwell Server |

### Driver Requirements
- NVIDIA GPU drivers compatible with the above GPUs
- See [NVIDIA Driver Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html) for specific versions

### Software Dependencies
- **GCC/G++**: Version 11 (for Ubuntu 24.04 compatibility)
- **Internet Access**: Required for downloading the Omniverse Kit SDK, extensions, and tools

### Installation Commands for Ubuntu
```bash
# Install build-essential
sudo apt-get install build-essential

# Install GCC/G++ 11
sudo apt-get install gcc-11 g++-11

# Set GCC/G++ 11 as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
```

## Isaac ROS 3.2 Package Installation Requirements

### ROS 2 Distribution Support
- ROS 2 Humble Hawksbill (Ubuntu 22.04)
- ROS 2 Jazzy (Ubuntu 24.04)

### Required ROS 2 Packages
- `rclpy`: Python client library for ROS 2
- `rclcpp`: C++ client library for ROS 2
- `ament_cmake`: Core CMake functions for building ROS 2 ament packages
- `ament_cmake_python`: For building and installing Python components within an ament package
- `ament_cmake_auto`: For automatically handling installation of directories

### Build System Requirements
- **CMake**: Minimum version 3.22.1
- **Python**: 3.8+ for ROS 2 compatibility
- **C++ Standard**: C++17 enforced for Isaac ROS packages

### Isaac ROS Package Structure
Isaac ROS packages typically include:
- Launch files in `launch` directory
- Parameter files in `params` directory
- Python scripts in `scripts` directory
- CMakeLists.txt with proper ROS 2 package configuration
- package.xml with dependencies and metadata