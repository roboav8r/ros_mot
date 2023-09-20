#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import time
import math

import torch
import gtsam
import numpy as np
from scipy.optimize import linear_sum_assignment

import rospy
import rospkg
import tf2_ros
import geometry_msgs

from tf2_geometry_msgs import PoseStamped
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# Detection object
class Detection():
    def __init__(self, ts, px, py, pz, prob, label):
        self.timestamp = ts
        # self.px, self.py, self.pz = px, py, pz
        self.pos = np.array([[px], [py], [pz]])
        self.prob = prob
        self.label = label
        # Can be empty/null/contain partial info
        # class conf % - float
        # ??? BB dimensions

# Track object
class Track():
    def __init__(self, det):
        self.timestamp = det.timestamp
        self.pos = det.pos
        self.vel = np.array([[0.],[0.],[0.]])
        self.prob = det.prob
        self.label = det.label
        self.vel_variance = 0.5

        # Kalman filter for this object
        self.kf = gtsam.KalmanFilter(6)
        self.state = self.kf.init(
            np.vstack((self.pos, self.vel)),
            np.diag([self.pos_variance,self.pos_variance,self.pos_variance,self.vel_variance,self.vel_variance,self.vel_variance])
            # TODO - set this to detection covariance
        )

        # Process model matrices
        self.proc_model = np.diag(np.ones(6))
        self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas([1,1,1,1,1,1])
    
    def compute_proc_model(self,dt):
        self.proc_model[0,3] = dt
        self.proc_model[1,4] = dt
        self.proc_model[2,5] = dt

    def compute_proc_noise(self,dt):
        # TODO - verify noise model coefficients
        self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas([0.25*self.vel_variance**4*dt**4,
                                                            0.25*self.vel_variance**4*dt**4,
                                                            0.25*self.vel_variance**4*dt**4,
                                                            0.5*self.vel_variance**2*dt**2,
                                                            0.5*self.vel_variance**2*dt**2,
                                                            0.5*self.vel_variance**2*dt**2])
    def predict(self, t):
        dt = (t - self.timestamp).to_sec()
        self.timestamp = t
        
        self.compute_proc_model(dt)
        self.compute_proc_noise(dt)
        self.state = self.kf.predict(self.state,self.proc_model,np.zeros((6,6)),np.zeros((6,1)),self.proc_noise)

# Graph Tracker object
class GraphTracker():
    def __init__(self, name, frame, pub):
        # Generic filter states
        self.name = name       
        self.init = False

        # ROS transforms
        self.frame_id = frame
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.pose_stamped = PoseStamped()

        # Track and detection parameters & variables
        self.detections = []
        self.tracks = []
        self.trk_delete_thresh = 0.25
        self.trk_delete_age = 10
        self.trk_delete_time = 3.
        self.trk_create_thresh = 0.75
        self.trk_create_age = 5
        self.trk_pub = pub

        # Assignment
        self.cost_matrix = np.array()

        # Initialize member variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.trk_msg = BoundingBoxArray()
    
    def delete_tracks(self):
        self.tracks = [track for track in self.tracks if track.prob < self.trk_delete_thresh]
        self.tracks = [track for track in self.tracks if track.age > self.trk_delete_age]

    def delete_track_candidates(self):
        self.trk_candidates = [track for track in self.trk_candidates if ((rospy.Time.now() - track.timestamp).to_sec() < self.trk_delete_time)]
   
    # Data association
    def cost(self, det, track):
        # Euclidean distance between positions
        return np.linalg.norm(det.pos - track.state.mean()[0:3])

    def compute_cost_matrix(self):
        self.cost_matrix = np.zeros((len(self.detections),len(self.tracks)))
        for ii,det in enumerate(self.detections):
            for jj,trk in enumerate(self.tracks):
                self.cost_matrix[ii,jj] = self.cost(det,trk)

    # Detection callback / main algorithm
    def det_callback(self, det_array_msg):
        
        # Populate detections list from detections message
        for det in det_array_msg.boxes:
            # self.pose_stamped = PoseStamped(det.header,det.pose)
            self.pose_stamped = self.tf_buf.transform(PoseStamped(det.header,det.pose), self.frame_id, rospy.Duration(1))
            self.detections.append(Detection(det.header.stamp, self.pose_stamped.pose.position.x, self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.z, det.value, det.label))

        # Propagate existing tracks
        for trk in self.tracks:
            trk.predict(det_array_msg.header.stamp)
        
        # Compute detection-track correspondences
        self.compute_cost_matrix()
        self.solve_cost_matrix()

        # TODO - Assign detections to track variables
        # Add factors, variables
        
        # TODO - Update tracks
        # for trk in self.tracks:
        #     trk.predict(det_array_msg.header.stamp)

        # Save unmatched detections as potential track candidates
        while self.detections:
            self.trk_candidates.append(Track(self.detections.pop()))
        
        # Delete tracks as needed
        self.delete_tracks()
        self.delete_track_candidates()

        # TODO - Convert tracks to track message
        # self.trk_msg.boxes = self.tracks

        # Publish tracks
        self.trk_msg.header = det_array_msg.header
        self.trk_pub.publish()

if __name__ == '__main__':
    # Initialize node
    rospy.init_node("tracker")
    print("Tracker node initialized")

    # Create publisher
    track_pub = rospy.Publisher("tracks_3d", BoundingBoxArray, queue_size=10)

    # Get parameters
    detector_dict = rospy.get_param("tracker")

    # Create detectors according to params
    for name,value in detector_dict.items():
        if value['type']=='graph_tracker':
            # Create tracker object
            tracker = GraphTracker(name, value['frame'], track_pub)

            # Create detector subscriber
            det_sub = rospy.Subscriber(value['det_topic'], BoundingBoxArray, tracker.det_callback)

    rospy.spin()