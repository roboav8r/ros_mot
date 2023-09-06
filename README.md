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
cd ~/my_ws/
rosdep install ros_mot
cd ~/my_ws/src/ros_mot
python3 -m pip install requirements.txt
```
### Alternative: CUDA not installed
```
cd ~/my_ws/src/ros_mot
python3 -m pip install requirements-no-cuda.txt
```
# Usage
## AB3DMOT
First, download the 3d detector model you'd like to use. For example:
```
roscd ros_mot
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest config/
```
Other options include:
'3dssd_4x4_kitti-3d-car', 'centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d', 'centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area1.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area2.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area3.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area4.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area5.py', 'dgcnn_4xb32-cosine-100e_s3dis-seg_test-area6.py', 'dv_second_secfpn_6x8_80e_kitti-3d-car', 'dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class', 'dv_pointpillars_secfpn_6x8_160e_kitti-3d-car', 'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune', 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'hv_pointpillars_regnet-1.6gf_fpn_sbn-all_free-anchor_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-3.2gf_fpn_head-free-anchor_sbn-all_8xb4-strong-aug-3x_nus-3d', 'groupfree3d_head-L6-O256_4xb8_scannet-seg.py', 'groupfree3d_head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O256_4xb8_scannet-seg.py', 'groupfree3d_w2x-head-L12-O512_4xb8_scannet-seg.py', 'h3dnet_3x8_scannet-3d-18class', 'imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class', 'imvotenet_stage2_16x8_sunrgbd-3d-10class', 'imvoxelnet_kitti-3d-car', 'monoflex_dla34_pytorch_dlaneck_gn-all_2x4_6x_kitti-mono3d', 'dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class', 'mask-rcnn_r50_fpn_1x_nuim', 'mask-rcnn_r50_fpn_coco-2x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_1x_nuim', 'mask-rcnn_r50_caffe_fpn_coco-3x_20e_nuim', 'mask-rcnn_r101_fpn_1x_nuim', 'mask-rcnn_x101_32x4d_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_1x_nuim', 'cascade-mask-rcnn_r50_fpn_coco-20e_20e_nuim', 'cascade-mask-rcnn_r101_fpn_1x_nuim', 'cascade-mask-rcnn_x101_32x4d_fpn_1x_nuim', 'htc_r50_fpn_coco-20e_1x_nuim', 'htc_r50_fpn_coco-20e_20e_nuim', 'htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim', 'paconv_ssg_8xb8-cosine-150e_s3dis-seg.py', 'paconv_ssg-cuda_8xb8-cosine-200e_s3dis-seg', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class', 'parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car', 'pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d', 'pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune', 'point-rcnn_8xb2_kitti-3d-3class', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg-xyz-only', 'pointnet2_ssg_2xb16-cosine-200e_scannet-seg', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg-xyz-only', 'pointnet2_msg_2xb16-cosine-250e_scannet-seg', 'pointnet2_ssg_2xb16-cosine-50e_s3dis-seg', 'pointnet2_msg_2xb16-cosine-80e_s3dis-seg', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class', 'pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_fpn_sbn-all_8xb4-amp-2x_nus-3d', 'pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymoD5-3d-3class', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-car', 'pointpillars_hv_secfpn_sbn_2x16_2x_waymo-3d-3class', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d', 'pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d', 'pointpillars_hv_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d', 'pointpillars_hv_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d', 'second_hv_secfpn_8xb6-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-80e_kitti-3d-3class', 'second_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-car', 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class', 'smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d', 'hv_ssn_secfpn_sbn-all_16xb2-2x_lyft-3d', 'hv_ssn_regnet-400mf_secfpn_sbn-all_16xb1-2x_lyft-3d', 'votenet_8xb16_sunrgbd-3d.py', 'votenet_8xb8_scannet-3d.py', 'votenet_iouloss_8x8_scannet-3d-18class', 'minkunet18_w16_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w20_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet18_w32_torchsparse_8xb2-amp-15e_semantickitti', 'minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti', 'minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti', 'cylinder3d_4xb4-3x_semantickitti', 'cylinder3d_8xb2-laser-polar-mix-3x_semantickitti', 'pv_rcnn_8xb2-80e_kitti-3d-3class', 'fcaf3d_2xb8_scannet-3d-18class', 'fcaf3d_2xb8_sunrgbd-3d-10class', 'fcaf3d_2xb8_s3dis-3d-5class', 'spvcnn_w16_8xb2-amp-15e_semantickitti', 'spvcnn_w20_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-15e_semantickitti', 'spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti'

# Acknowledgements
This repository is intended to develop, implement, and/or evaluate multiobject tracking capabilities on ROS robots and systems. While working on my PhD at UT Austin, I realized that there was a lack of accessible and easily usable ROS packages for MOT, which motivated me to create this.

This effort began with Xinshuo Weng et al's [3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)](https://github.com/xinshuoweng/AB3DMOT), which--as far as I can tell--was the first package in recent history designed for use *specifically* on real-time robotic systems. From there, a [ROS implementation](https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT) was created. I am not affiliated with the authors of either package, but I used their work as a starting point for this repository. It is my goal to make their work fit within a more general and modular framework, and include additional MOT algorithms and techniques for other applications.