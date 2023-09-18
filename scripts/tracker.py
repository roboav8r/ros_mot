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

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# Graph Tracker object
class GraphTracker():
    def __init__(self, name, pub):
        # Member variables
        self.name = name
        self.init = False
        self.detections = []
        self.tracks = []
        self.graph = gtsam.NonlinearFactorGraph()
        self.trk_delete_thresh = 0.25
        self.trk_delete_age = 10
        self.trk_create_thresh = 0.75
        self.trk_create_age = 5
        self.trk_pub = pub
        self.trk_msg = BoundingBoxArray()
    
    def delete_tracks(self):
        self.tracks = [track for track in self.tracks if track.prob < self.trk_delete_thresh]
        self.tracks = [track for track in self.tracks if track.age > self.trk_delete_age]

    def det_callback(self, det_array_msg):
        self.detections = det_array_msg.boxes
        
        # TODO - Compute detection-track correspondences
        
        # TODO - Assign detections to track variables
        
        # TODO - create tracks as needed
        
        # Delete tracks as needed
        self.delete_tracks()

        # Publish tracks
        self.trk_msg.header = det_array_msg.header
        self.trk_msg.boxes = self.tracks
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
            tracker = GraphTracker(name, track_pub)

            # Create detector subscriber
            det_sub = rospy.Subscriber(value['det_topic'], BoundingBoxArray, tracker.det_callback)

    rospy.spin()