# ros_mot
Repo for implementing multi-object tracking (MOT) algorithms in Robot Operating System (ROS)

# Installation

## Prerequisites
### Git Large File Storage (LFS)
This repo uses Git LFS for large files like rosbags and models. To set up Git LFS on your system, use the instructions here:
https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md 
### CUDA
This assumes you have CUDA [11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) installed. If you do not have CUDA installed, follow the instructions here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Be sure to add CUDA's `bin` directory to your `$PATH` variable.

## Clone the repository
```
cd ~/my_ws/src
git clone --recursive https://github.com/roboav8r/ros_mot
```

## Install dependencies
```
# Python dependencies
cd ~/my_ws/src/ros_mot
python3 -m pip install -r requirements.txt
```

## Build
```
cd ~/my_ws/
catkin build ros_mot
source devel/setup.bash
rosdep install ros_mot
```

# Usage
## Detectors
This package uses [`mmdetection3d`](https://github.com/open-mmlab/mmdetection3d) for 3d detections. `mmdetection3d` is installed via `pip`, and provides a common interface for training and running common 3D detectors (e.g. CenterPoint, pointpillars) trained on common datasets (KITTI, nuScenes). Note that the `mim` command is available through `openmim`, which is listed in `requirements.txt` and installed in the steps above.

To use, first download the 3d detector model you'd like to use. For example:
```
roscd ros_mot
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest models/
# Or for an indoor model
mim download mmdet3d --config fcaf3d_2xb8_sunrgbd-3d-10class --dest models/
```
For more 3D detector options, see the [mmdetection3d model zoo](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/model_zoo.md).

## Trackers
### AB3DMOT
### SimpleTrack
# Miscellaneous
This package includes everything you might need to get started, but here are some additional resources.
## Datasets
- KITTI: https://www.cvlibs.net/datasets/kitti/user_register.php

# Acknowledgements
This repository is intended to develop, implement, and/or evaluate multiobject tracking capabilities on ROS robots and systems. While working on my PhD at UT Austin, I realized that there was a lack of accessible and easily usable ROS packages for MOT, which motivated me to create this.

This effort began with Xinshuo Weng et al's [3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)](https://github.com/xinshuoweng/AB3DMOT), which--as far as I can tell--was the first package in recent history designed for use *specifically* on real-time robotic systems. From there, a [ROS implementation](https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT) was created. I am not affiliated with the authors of either package, but I used their work as a starting point for this repository. It is my goal to make their work fit within a more general and modular framework, and include additional MOT algorithms and techniques for other applications.