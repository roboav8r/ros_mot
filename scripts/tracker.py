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

# Sensor object
class SensorModel():
    # TODO - (modularity/usability) accommodate multiple sensor types and models
    def __init__(self):
        self.obs_model = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
        self.obs_noise = gtsam.noiseModel.Diagonal.Sigmas([.1,.1,.1])
        

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
        # TODO - (cleanup) remove pos,vel and just use self.state
        # TODO - convert det into tracker frame
        self.timestamp = det.timestamp
        self.missed_det = 0
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
        # TODO (accuracy) - verify noise model coefficients
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

    def update(self, det, sensor_mdl):
        # TODO (cleanup) - remove pos,vel and just use self.state
        self.state = self.kf.update(self.state, sensor_mdl.obs_model, det.pos, sensor_mdl.obs_noise)
        self.timestamp = det.timestamp
        self.pos = self.state.mean()[0:3]
        self.vel = self.state.mean()[3:6]
        self.missed_det = 0

# Graph Tracker object
class Tracker():
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
        self.asgn_thresh = 4.
        self.cost_matrix = np.empty(0)
        self.det_asgn_idx = [] 
        self.trk_asgn_idx = []

        # Initialize member variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.box_msg = BoundingBox()
        self.trk_msg = BoundingBoxArray()
    
    def delete_tracks(self):
        self.tracks = [track for track in self.tracks if track.prob > self.trk_delete_thresh]
        self.tracks = [track for track in self.tracks if track.age < self.trk_delete_age]
        self.tracks = [track for track in self.tracks if ((rospy.Time.now() - track.timestamp) < self.trk_delete_time)]

    # Data association
    def cost(self, det, track):
        # Euclidean distance between positions
        return np.linalg.norm(det.pos - track.state.mean()[0:3])

    def compute_cost_matrix(self):
        self.cost_matrix = np.zeros((len(self.detections),len(self.tracks)))
        for ii,det in enumerate(self.detections):
            for jj,trk in enumerate(self.tracks):
                self.cost_matrix[ii,jj] = self.cost(det,trk)
    
    def solve_cost_matrix(self):
        self.det_asgn_idx, self.trk_asgn_idx = linear_sum_assignment(self.cost_matrix)
        self.det_asgn_idx, self.trk_asgn_idx = list(self.det_asgn_idx), list(self.trk_asgn_idx)
        
        # If cost above threshold, remove the match
        assert(len(self.det_asgn_idx) == len(self.trk_asgn_idx))
        ii = len(self.det_asgn_idx)
        while ii:
            idx = ii-1
            if self.cost_matrix[self.det_asgn_idx[idx],self.trk_asgn_idx[idx]] > self.asgn_thresh:
                print(idx)
                print('too high')
                del self.det_asgn_idx[idx], self.trk_asgn_idx[idx]       
            ii -=1
        assert(len(self.det_asgn_idx) == len(self.trk_asgn_idx))

    def format_trk_msg(self, det_array_msg):
        self.trk_msg = BoundingBoxArray()
        self.trk_msg.header.stamp = det_array_msg.header.stamp
        self.trk_msg.header.frame_id = self.frame_id

        print("PUBLISHING/OUTPUT")
        print("Num. tracks:")
        print(len(self.tracks) + "\n")
        for trk in self.tracks:
            self.box_msg = BoundingBox()
            self.box_msg.header = self.trk_msg.header
            self.box_msg.pose.position.x = trk.state.mean()[0]
            self.box_msg.pose.position.y = trk.state.mean()[1]
            self.box_msg.pose.position.z = trk.state.mean()[2]
            # TODO (visualization/accuracy) - update quaternion, bbox dimensions
            self.box_msg.pose.orientation.z = 1
            self.box_msg.dimensions.x = 0.25
            self.box_msg.dimensions.y = 0.25
            self.box_msg.dimensions.z = 1
            self.box_msg.value = trk.prob
            self.box_msg.label = trk.label
            self.trk_msg.boxes.append(self.box_msg)


    # Detection callback / main algorithm
    def det_callback(self, det_array_msg, sensor_mdl):
        
        # Populate detections list from detections message
        for det in det_array_msg.boxes:
            # Convert to tracker frame and add to detections
            self.pose_stamped = self.tf_buf.transform(PoseStamped(det.header,det.pose), self.frame_id, rospy.Duration(1))
            self.detections.append(Detection(det.header.stamp, self.pose_stamped.pose.position.x, self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.z, det.value, det.label))

        # Propagate existing tracks
        for trk in self.tracks:
            trk.predict(det_array_msg.header.stamp)
        
        # Compute detection-track assignments
        self.compute_cost_matrix()
        self.solve_cost_matrix()

        # Update matched tracks with matched detections
        # TODO (improvement) - add factors/variables to graph as appropriate
        for det_idx, trk_idx in zip(self.det_asgn_idx, self.trk_asgn_idx):
            self.tracks[trk_idx].update(self.detections[det_idx], sensor_mdl)

        # Handle unmatched tracks
        # TODO (improvement) - get subset of list 
        for i, trk in enumerate(self.tracks):
            if i not in self.trk_asgn_idx: # If track is unmatched
                trk.missed_det +=1 # Increment missed detection counter

        # Handle unmatched detections / initialize new tracks
        while self.detections:
            if (len(self.detections)-1) in self.det_asgn_idx: # If last detection is unmatched
                self.tracks.append(Track(self.detections.pop())) # Create a new track
            else: self.detections.pop() # Otherwise remove it

        # Delete tracks as needed
        self.delete_tracks()

        # Convert tracks to track message
        self.format_trk_msg(det_array_msg)

        # Publish tracks
        self.trk_pub.publish(self.trk_msg)

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
            tracker = Tracker(name, value['frame'], track_pub)

            # Create detector subscriber
            # TODO (usability) - read in from config file
            oakd_model = SensorModel()
            det_sub = rospy.Subscriber(value['det_topic'], BoundingBoxArray, tracker.det_callback, oakd_model)

    rospy.spin()