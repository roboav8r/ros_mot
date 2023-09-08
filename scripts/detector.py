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

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures.points import BasePoints

# Constants / parameters
callback_map = {'PointCloud2': 'self.pc2_callback'} # message type -> callback
# publisher_map = {'PointCloud2': 'self.bbarray_pub'} #

# Base detector class
class mmdetector3d():			  	
	def __init__(self, name, model, topic, msg_type, pub, viz):                      
		print('Starting detector: ', name)
		self.name = name
		self.model = model
		self.topic = topic
		self.msg_type = msg_type
		self.viz = viz

		# Create subscriber
		self.sub = rospy.Subscriber(self.topic, eval(self.msg_type), eval(callback_map[self.msg_type]))

		# Create publisher
		self.pub = pub

		# Create empty messages
		self.bb_array_msg = BoundingBoxArray()
		self.bb_msg = BoundingBox()

	def pc2_callback(self, pc2_msg):
		# Format ros pc2 message -> mmdet3d BasePoints
		pc_list = point_cloud2.read_points_list(pc2_msg)
		pc_np = np.array(list(pc_list))
		result, _  = inference_detector(self.model, pc_np)

		# Clear messages
		self.bb_array_msg = BoundingBoxArray()
		self.bb_msg = BoundingBox()

		# Populate messages
		# TODO 

		# Publish messages
		self.pub.publish(self.bb_array_msg)

if __name__ == '__main__':

	# Initialize node
	rospy.init_node("detector")
	print("Detector node initialized")

	# Create common publishers
	detection3d_pub = rospy.Publisher("detections_3d", BoundingBoxArray, queue_size=10)

	# Get package path
	pkg_path = rospkg.RosPack().get_path('ros_mot')

	# Get Torch device
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Get parameters
	detector_dict = rospy.get_param("detectors")
	viz = rospy.get_param("~visualization")
	
	# Create detectors according to params
	for name,value in detector_dict.items():

		if value['type']=='mmdet3d':
			# Create mmdet3d model
			cfg_file = os.path.join(pkg_path,value['config'])
			ckpt_file = os.path.join(pkg_path,value['checkpoint'])
			model = init_model(cfg_file, ckpt_file, device)

			# Create detector object
			mmdetector3d(name,model,value['topic'],value['msg_type'],detection3d_pub, viz)

	rospy.spin()