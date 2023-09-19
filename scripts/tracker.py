#!/usr/bin/env python3

from __future__ import print_function
import numpy as np, copy, math, sys, argparse

import os
import sys
import rospy
import torch
import time
import rospkg
import math

import gtsam

import tf2_ros

import geometry_msgs
from tf2_geometry_msgs import PoseStamped
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# Detection object
class Detection():
    def __init__(self, ts, px, py, pz, prob, label):
        self.timestamp = ts
        self.px, self.py, self.pz = px, py, pz
        self.prob = prob
        self.label = label
        # Can be empty/null/contain partial info
        # class conf % - float
        # ??? BB dimensions

# Track object
class Track():
    def __init__(self, det):
        self.timestamp = det.timestamp
        self.px = det.px
        self.py = det.py
        self.pz = det.pz
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.prob = det.prob
        self.label = det.label

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

        # Track and detection parameters
        self.detections = []
        self.trk_candidates = []
        self.tracks = []
        self.trk_delete_thresh = 0.25
        self.trk_delete_age = 10
        self.trk_delete_time = 3.
        self.trk_create_thresh = 0.75
        self.trk_create_age = 5
        self.trk_pub = pub

        # Initialize member variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.trk_msg = BoundingBoxArray()
    
    def delete_tracks(self):
        self.tracks = [track for track in self.tracks if track.prob < self.trk_delete_thresh]
        self.tracks = [track for track in self.tracks if track.age > self.trk_delete_age]

    def delete_track_candidates(self):
        self.trk_candidates = [track for track in self.trk_candidates if ((rospy.Time.now() - track.timestamp).to_sec() < self.trk_delete_time)]
        # for track in self.trk_candidates:
        #     print((rospy.Time.now() - track.timestamp).to_sec())

    def det_callback(self, det_array_msg):
        
        # Populate detections list from detections message
        for det in det_array_msg.boxes:
            # self.pose_stamped = PoseStamped(det.header,det.pose)
            self.pose_stamped = self.tf_buf.transform(PoseStamped(det.header,det.pose), self.frame_id, rospy.Duration(1))
            self.detections.append(Detection(det.header.stamp, self.pose_stamped.pose.position.x, self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.z, det.value, det.label))

        # TODO - propagate existing tracks and candidates     
        
        # TODO - Compute detection-track correspondences
        
        # TODO - Assign detections to track variables
        # Add factors, variables
        # 
        
        # Save unmatched detections as potential track candidates
        while self.detections:
            self.trk_candidates.append(Track(self.detections.pop()))

        # TODO - create tracks from candidates

        # TODO - Populate and solve graph

        
        # Delete tracks as needed
        self.delete_tracks()
        self.delete_track_candidates()

        print(self.detections)
        print(self.trk_candidates)
        print()

        # Publish tracks
        self.trk_msg.header = det_array_msg.header
        self.trk_msg.boxes = self.tracks
        self.trk_pub.publish()

# Helper functions
def compute_similarity(det, track):
    

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