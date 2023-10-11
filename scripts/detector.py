#!/usr/bin/env python

# Imports 
import numpy as np

import rospy
import rospkg

from tf import transformations as tf_trans

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from depthai_ros_msgs.msg import SpatialDetectionArray
from tracking_msgs.msg import DetectedObject, DetectedObjects
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

# from mmdet3d.apis import init_model, inference_detector
# from mmdet3d.structures.points import BasePoints
# from mmdet3d.structures.det3d_data_sample import Det3DDataSample

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils
# from pcdet.datasets.coda import coda_utils

# Constants / parameters
callback_map = {'PointCloud2': 'self.pc2_callback',
				'SpatialDetectionArray': 'self.oakd_callback'} # message type -> callback

# # mmdetector3d class
# class mmdetector3d():			  	
# 	def __init__(self, name, model, topic, msg_type, conf_thresh, labels, pub, viz):                      
# 		print('Starting detector: ', name)
# 		self.name = name
# 		self.model = model
# 		self.topic = topic
# 		self.msg_type = msg_type
# 		self.viz = viz
# 		self.conf_thresh = conf_thresh
# 		self.cat_labels = labels

# 		# Create subscriber
# 		self.sub = rospy.Subscriber(self.topic, eval(self.msg_type), eval(callback_map[self.msg_type]))

# 		# Create publishers
# 		self.pub = pub

# 		# Create empty messages
# 		self.bb_array_msg = BoundingBoxArray()
# 		self.bb_msg = BoundingBox()

# 		# Initialize empty data structures
# 		self.pc_list = []
# 		self.pc_np = np.zeros((100000,4),dtype=np.float64)
# 		self.pc_tensor = torch.from_numpy(self.pc_np).to(device)
# 		self.result = Det3DDataSample()
# 		self.lidar_msg = PointCloud2()

# 	def format_pc2_msg(self):
# 		self.bb_array_msg = BoundingBoxArray()
# 		self.bb_array_msg.header = self.lidar_msg.header
		
# 		for ii in range(len(self.result.pred_instances_3d.scores_3d)):
# 			if self.result.pred_instances_3d.scores_3d[ii] > self.conf_thresh:
# 				self.bb_msg = BoundingBox()
# 				self.bb_msg.header = self.lidar_msg.header
# 				self.bb_msg.value = self.result.pred_instances_3d.scores_3d[ii]
# 				self.bb_msg.label = self.result.pred_instances_3d.labels_3d[ii]
# 				self.bb_msg.pose.position.x = self.result.pred_instances_3d.bboxes_3d.tensor[ii,0].float()
# 				self.bb_msg.pose.position.y = self.result.pred_instances_3d.bboxes_3d.tensor[ii,1].float()
# 				self.bb_msg.pose.position.z = self.result.pred_instances_3d.bboxes_3d.tensor[ii,2].float()
# 				self.bb_msg.pose.orientation.x,self.bb_msg.pose.orientation.y,self.bb_msg.pose.orientation.z,self.bb_msg.pose.orientation.w   = tf_trans.quaternion_from_euler(0,0,self.result.pred_instances_3d.bboxes_3d.tensor[ii,6].float())
# 				self.bb_msg.dimensions.x = self.result.pred_instances_3d.bboxes_3d.tensor[ii,3].float()
# 				self.bb_msg.dimensions.y = self.result.pred_instances_3d.bboxes_3d.tensor[ii,4].float()
# 				self.bb_msg.dimensions.z = self.result.pred_instances_3d.bboxes_3d.tensor[ii,5].float()
# 				self.bb_array_msg.boxes.append(self.bb_msg)

# 	def pc2_callback(self, pc2_msg):
# 		# Format ros pc2 message -> mmdet3d BasePoints
# 		self.lidar_msg = pc2_msg
# 		self.pc_list = point_cloud2.read_points_list(pc2_msg)
# 		self.pc_np = np.array(list(self.pc_list))
# 		self.result, _  = inference_detector(self.model, self.pc_np)

# 		self.format_pc2_msg()

# 		# Publish messages
# 		self.pub.publish(self.bb_array_msg)

# OAK-D detector class
class oakd_detector():			  	
	def __init__(self, name, values, pub):                      
		print('Starting detector: ', name)
		self.name = name
		self.topic = values['topic']
		self.msg_type = 'SpatialDetectionArray'
		# self.conf_thresh = conf_thresh
		self.cat_labels = values['labels']
		# self.hfov_atan = math.atan(hfov*math.pi/180)
		# self.vfov_atan = math.atan(vfov*math.pi/180)
		# self.height = img_height
		# self.width = img_width
		self.det_id_count = 0
		self.variance = np.array(values['sensor_variance'])
		self.covariance = np.array([[self.variance[0]**2, 0., 0.],
					 	   [0., self.variance[1]**2, 0.],
						   [0., 0., self.variance[2]**2]])

		print(self.variance)
		print(self.covariance)

		# Create subscriber
		self.sub = rospy.Subscriber(self.topic, eval(self.msg_type), eval(callback_map[self.msg_type]))

		# Create publishers
		self.pub = pub

		# Initialize empty data structures
		self.oakd_msg = SpatialDetectionArray()

		# Create empty messages
		self.det_msg = DetectedObject()
		self.det_msgs = DetectedObjects()

	def format_det_msg(self):
		self.det_msgs = DetectedObjects()
		self.det_msgs.header = self.oakd_msg.header
		self.det_msgs.sensor_name = self.name
		
		for ii in range(len(self.oakd_msg.detections)):
			if self.oakd_msg.detections[ii].results[0].score > self.conf_thresh:
				self.det_msg = DetectedObject()
				self.det_msg.detection_id = self.det_id_count
				self.det_msg.detection_type = 'pos'
				self.det_msg.class_id = self.oakd_msg.detections[ii].results[0].id
				self.det_msg.class_confidence = self.oakd_msg.detections[ii].results[0].score
				self.det_msg.class_string = self.cat_labels[self.det_msg.class_id]
				self.det_msg.pose.pose.position.x = self.oakd_msg.detections[ii].position.x
				self.det_msg.pose.pose.position.y = -self.oakd_msg.detections[ii].position.y # Fix OAK-D's left hand coords
				self.det_msg.pose.pose.position.z = self.oakd_msg.detections[ii].position.z
				self.det_msg.pose.pose.orientation.x,self.det_msg.pose.pose.orientation.y,self.det_msg.pose.pose.orientation.z,self.det_msg.pose.pose.orientation.w = 0,0,0,1
				self.det_msg.pose.covariance = self.covariance

				self.det_id_count+=1
				self.det_msgs.detections.append(self.det_msg)


	def oakd_callback(self, msg):
		# Receive ROS message, format, and publish as tracking_msgs/DetectedObjects message
		self.oakd_msg = msg
		self.format_det_msg()
		self.pub.publish(self.det_msgs)

# class pcdet_detector:
# 	def __init__(self, name, topic, msg_type, cfg_file, pub):
# 		print('Starting detector: ', name)
# 		self.name = name
# 		self.topic = topic
# 		self.msg_type = msg_type
# 		self.cfg = cfg

# 		# Create subscriber
# 		self.sub = rospy.Subscriber(self.topic, eval(self.msg_type), eval(callback_map[self.msg_type]))

# 		# Create publishers
# 		self.pub = pub

# 		# Create empty messages
# 		self.bb_array_msg = BoundingBoxArray()
# 		self.bb_msg = BoundingBox()

# 		# Create dummy dataset

# 		# Create detector model
# 		cfg_from_yaml_file(cfg_file, self.cfg)
# 		self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=dummy_dataset)
# 		self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
# 		self.model.cuda()
# 		self.model.eval()

# 	def format_pc2_msg(self):
# 		self.bb_array_msg = BoundingBoxArray()
# 		self.bb_array_msg.header = self.lidar_msg.header
		
# 		for ii in range(len(self.result.pred_instances_3d.scores_3d)):
# 			if self.result.pred_instances_3d.scores_3d[ii] > self.conf_thresh:
# 				self.bb_msg = BoundingBox()
# 				self.bb_msg.header = self.lidar_msg.header
# 				self.bb_msg.value = self.result.pred_instances_3d.scores_3d[ii]
# 				self.bb_msg.label = self.result.pred_instances_3d.labels_3d[ii]
# 				self.bb_msg.pose.position.x = self.result.pred_instances_3d.bboxes_3d.tensor[ii,0].float()
# 				self.bb_msg.pose.position.y = self.result.pred_instances_3d.bboxes_3d.tensor[ii,1].float()
# 				self.bb_msg.pose.position.z = self.result.pred_instances_3d.bboxes_3d.tensor[ii,2].float()
# 				self.bb_msg.pose.orientation.x,self.bb_msg.pose.orientation.y,self.bb_msg.pose.orientation.z,self.bb_msg.pose.orientation.w   = tf_trans.quaternion_from_euler(0,0,self.result.pred_instances_3d.bboxes_3d.tensor[ii,6].float())
# 				self.bb_msg.dimensions.x = self.result.pred_instances_3d.bboxes_3d.tensor[ii,3].float()
# 				self.bb_msg.dimensions.y = self.result.pred_instances_3d.bboxes_3d.tensor[ii,4].float()
# 				self.bb_msg.dimensions.z = self.result.pred_instances_3d.bboxes_3d.tensor[ii,5].float()
# 				self.bb_array_msg.boxes.append(self.bb_msg)

# 	def pc2_callback(self, pc2_msg):
# 		# Format ros pc2 message
# 		self.lidar_msg = pc2_msg
# 		self.pc_list = point_cloud2.read_points_list(pc2_msg)
# 		self.pc_np = np.array(list(self.pc_list))
# 		self.result, _  = inference_detector(self.model, self.pc_np)

# 		self.format_pc2_msg()

# 		# Publish messages
# 		self.pub.publish(self.bb_array_msg)

if __name__ == '__main__':

	# Initialize node
	rospy.init_node("detector_node")
	print("Detector node initialized")

	# Create common publishers
	detection_pub = rospy.Publisher("detected_objects", DetectedObjects, queue_size=10)

	# Get package path
	pkg_path = rospkg.RosPack().get_path('ros_mot')

	# Get Torch device
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Get parameters
	detector_dict = rospy.get_param("detectors")
	
	# Create detectors according to params
	for detector_name, detector_params in detector_dict.items():

		if detector_params['type']=='oakd':
			# Create OAK-D detector object
			oakd_detector(detector_name, detector_params, detection_pub)

		# if value['type']=='mmdet3d':
		# 	# Create mmdet3d model
		# 	cfg_file = os.path.join(pkg_path,value['config'])
		# 	ckpt_file = os.path.join(pkg_path,value['checkpoint'])
		# 	model = init_model(cfg_file, ckpt_file, device)

		# 	# Create detector object
		# 	mmdetector3d(name,model,value['topic'],value['msg_type'], value['conf_thresh'], value['cat_labels'], detection_pub, viz)

		# if value['type']=='pcdet':
		# 	cfg_file = os.path.join(pkg_path,value['config'])
		# 	ckpt_file = os.path.join(pkg_path,value['checkpoint'])
		# 	pcdet_detector(name, value['topic'],value['msg_type'], cfg_file, detection_pub)

	rospy.spin()