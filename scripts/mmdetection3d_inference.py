#!/usr/bin/env python

# Original Author of AB3DMOT repo: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# Modified for ROS by: Pardis Taghavi
# email: taghavi.pardis@gmail.com

from __future__ import print_function
import numpy as np, copy, math, sys, argparse

import os
import sys
import rospy
import torch
import time
import rospkg

# Get the directory path of the current file
# current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the desired path relative to the current file's location
# libs_dir = os.path.join(current_dir, "AB3DMOT/AB3DMOT_libs")
# xinshuo_lib=os.path.join(current_dir,"AB3DMOT/Xinshuo_PyToolbox")
# Add the path to sys.path
# sys.path.append(libs_dir)
# sys.path.append(xinshuo_lib)



from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.vis import vis_obj

from ab3dmot_ros.kalman_filter import KF
from ab3dmot_ros.kitti_oxts import get_ego_traj, egomotion_compensation_ID
from ab3dmot_ros.kitti_oxts import load_oxts ,_poses_from_oxts
from ab3dmot_ros.kitti_calib import Calibration
from ab3dmot_ros.model import AB3DMOT
from ab3dmot_ros.box import Box3D


#import std_msgs.msg
from mmdet3d.apis import init_model, inference_detector
# from mmdet3d.core.points import get_points_type
from mmdet3d.structures.points import BasePoints
#from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R
import ros_numpy
from collections import namedtuple
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix#, Image, CameraInfo
from geometry_msgs.msg import TwistStamped, Quaternion
import message_filters
#import torchvision.transforms as transforms
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
#from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray, VisionInfo
from visualization_msgs.msg import Marker,MarkerArray
lim_x=[0, 50]
lim_y=[-20,20]
lim_z=[-3,3]
np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class detector_3d():			  	
	def __init__(self, model=None, pointcloud_topic=None):                    

		self.cats=["Pedestrian", "Cyclist", "Car"]
		
		#self.vis = cfg.vis
		self.model=model
		self.vis_dir = False
		self.vis = False
		self.dets=[]
		#self.oxts_packets=[]
		self.affi_process =True	# post-processing affinity
		self.dataset="KITTI"
		self.det_name='pointrcnn'
		self.debug_id = None
		self.ID_start=1 
		torch.set_num_threads(4)
		self.pcdSub=rospy.Subscriber(pointcloud_topic, PointCloud2, self.callbackFunction)
		self.marker_pub = rospy.Publisher( 'visualization_marker_array', MarkerArray)
		self.bbox_publish = rospy.Publisher("tracking_bboxes", BoundingBoxArray, queue_size=10)

		# counter
		self.trackers = []
		self.frame_count = 0
		self.ID_count = [0]
		self.id_now_output = []
		self.oxts_packets = []

		#cfg file
		cfg=namedtuple('cfg', ['description','speed','save_root', 'dataset','split','det_name', 'cat_list',
			'score_threshold', 'num_hypo', 'ego_com','vis','affi_pro'])
		cfg.description='AB3DMOT'
		cfg.speed=1
		cfg.dataset='KITTI'
		cfg.split='val'
		cfg.det_name='pointrcnn'
		cfg.cat_list=['Car', 'Pedestrian', 'Cyclist']
		cfg.score_threshold=-10000 # can be changed
		cfg.num_hypo=1
		cfg.ego_com=False
		cfg.vis=False
		cfg.affi_pro=True
		
		# config
		self.cat = "Car"
		self.ego_com = cfg.ego_com 		# ego motion compensation'
		self.affi_process = cfg.affi_pro	# post-processing affinity
		self.get_param(cfg, self.cat)

		rospy.spin()

	def get_param(self, cfg, cat):
		# get parameters for each dataset

		if cfg.dataset == 'KITTI':
			if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'pointrcnn':			# tuned for PointRCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':			
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		
		else: assert False, 'no such dataset'

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1	
		self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
			algm, metric, thres, max_age, min_hits

		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

	
	def crop_pointcloud(self, pointcloud):

		mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >= lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
		pointcloud = pointcloud[mask]
		return pointcloud

	def callbackFunction(self, lidarMsg):
		start=rospy.Time.now().to_sec()
		start1=rospy.Time.now().to_sec()

		pc = ros_numpy.numpify(lidarMsg)
		points = np.column_stack((pc['x'], pc['y'],pc['z'], pc['i']))

		pc_arr=self.crop_pointcloud((points)) #reduce computational expense
		pointcloud_np = BasePoints(pc_arr, points_dim=pc_arr.shape[-1], attribute_dims=None)
		result, _  = inference_detector(self.model, pointcloud_np)

		#detections
		box = result[0]['boxes_3d'].tensor.numpy()
		scores = result[0]['scores_3d'].numpy()
		label = result[0]['labels_3d'].numpy()

		# box format : [ x, y, z, xdim(l), ydim(w), zdim(h), orientation] + label score
		# dets format : hwlxyzo + class
		dets=box[ :, [5,4,3,0,1,2,6]]
		info_data=[]
		dic_dets={}
		info_data = np.stack((label, scores), axis=1)
	
		dic_dets={'dets': dets, 'info': info_data}

		start=rospy.Time.now().to_sec()
	
		cat_res = dic_dets

		self.ID_start=max(self.ID_start, self.ID_count[0]) ##global counter
		trk_result=cat_res[0]
		#print("*******", trk_result)
		end=rospy.Time.now().to_sec()
		print("time for tracking",(end-start))

		bbox_array=BoundingBoxArray()
		cats=["Pedestrian", "Cyclist", "Car"]
		idx = 0
		self.markerArray = MarkerArray()

		for i, trk in enumerate(trk_result):
			if np.size(trk) == 0:
				continue
			if trk[9]>0.5:
				q = yaw_to_quaternion(trk[6])
				bbox = BoundingBox()
				marker = Marker()

				marker.header = lidarMsg.header
				marker.type = marker.TEXT_VIEW_FACING
				marker.id = int(trk[7])
				marker.text = f"{int(trk[7])} {cats[int(trk[8])]}"
				marker.action = marker.ADD
				marker.frame_locked = True
				marker.lifetime = rospy.Duration(0.1)
				marker.scale.x, marker.scale.y,marker.scale.z = 0.8, 0.8, 0.8
				marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 1.0, 1.0
				marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = trk[3], trk[4], trk[5] + 2
				
				bbox.header = lidarMsg.header #.seq = int(trk[7])
				#bbox.header.stamp = lidarMsg.header.stamp
				#bbox.header.frame_id = lidarMsg.header.frame_id
				bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z = trk[3], trk[4], trk[5]
				bbox.pose.orientation.w, bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z = q[3], q[0], q[1], q[2]
				bbox.dimensions.x, bbox.dimensions.y, bbox.dimensions.z = trk[2], trk[1], trk[0]
				bbox.value = trk[9]
				bbox.label = int(trk[8])
				bbox_array.header = bbox.header
				bbox_array.boxes.append(bbox)
				self.markerArray.markers.append(marker)
		
		#bbox_array.header.frame_id = lidarMsg.header.frame_id
		#print("len of bbox array from tracking", len(bbox_array.boxes))
		#print(bbox_array.boxes)
		if len(bbox_array.boxes) != 0:
			self.bbox_publish.publish(bbox_array)
			self.marker_pub.publish(self.markerArray)
			bbox_array.boxes = []

		else:
			bbox_array.boxes = []
			self.bbox_publish.publish(bbox_array)
		end1=rospy.Time.now().to_sec()
		print("time for publishing",(end1-start1))


if __name__ == '__main__':
	rospy.init_node("lidar_detector")
	print("3D lidar detector node initialized")
	
	rospack = rospkg.RosPack()
	pkg_dir = rospack.get_path('ros_mot')
	config_file = os.path.join(pkg_dir,'config','pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py')
	checkpoint_file =os.path.join(pkg_dir,'config','hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth')
	device= torch.device('cuda:0')
	topic="/kitti/velo/pointcloud"

	model = init_model(config_file, checkpoint_file, device)

	detector_3d(model=model, pointcloud_topic=topic)
