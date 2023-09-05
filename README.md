# ros_mot
Repo for implementing multi-object tracking (MOT) algorithms in Robot Operating System (ROS)

# Installation

## Prerequisites
This assumes you have CUDA [11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) installed. If you do not have CUDA installed, follow the instructions here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

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

# Acknowledgements
This repository is intended to develop, implement, and/or evaluate multiobject tracking capabilities on ROS robots and systems. While working on my PhD at UT Austin, I realized that there was a lack of accessible and easily usable ROS packages for MOT, which motivated me to create this.

This effort began with Xinshuo Weng et al's [3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)](https://github.com/xinshuoweng/AB3DMOT), which--as far as I can tell--was the first package in recent history designed for use *specifically* on real-time robotic systems. From there, a [ROS implementation](https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT) was created. I am not affiliated with the authors of either package, but I used their work as a starting point for this repository. It is my goal to make their work fit within a more general and modular framework, and include additional MOT algorithms and techniques for other applications.