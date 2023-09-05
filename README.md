# ros_mot
Repo for implementing multi-object tracking (MOT) algorithms in Robot Operating System (ROS)

# Installation

## Prerequisites
This assumes you have CUDA installed. If you do not have CUDA installed, follow the instructions here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Be sure to add CUDA's `bin` directory to your `$PATH` variable.

To run without CUDA, use the alternate dependency file as specified below.
## Clone the repository
```
cd ~/my_ws/src
git clone --recursive https://github.com/roboav8r/ros_mot
```

## Install dependencies
```
cd ~/my_ws/src/ros_mot
python3 -m pip install requirements.txt
```
### Alternative: CUDA not installed
```
cd ~/my_ws/src/ros_mot
python3 -m pip install requirements-no-cuda.txt
```