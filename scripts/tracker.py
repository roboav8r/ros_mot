#!/usr/bin/env python3

import gtsam
import numpy as np
from scipy.optimize import linear_sum_assignment

import rospy
import tf2_ros

from tf2_geometry_msgs import PoseStamped
from visualization_msgs.msg import MarkerArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# Sensor object
class SensorModel():
    # TODO - (modularity/usability) accommodate multiple sensor types and models
    def __init__(self):
        self.obs_model = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
        self.obs_noise = gtsam.noiseModel.Diagonal.Sigmas([.05,.05,.1]) # Used for recurring updates
        self.obs_cov = np.diag([.05,.05,.1, .1, .1, .1]) # Used for initial detection cov estimate

# Detection object
class Detection():
    def __init__(self, ts, px, py, pz, bbx, bby, bbz, prob, label, transform):
        self.timestamp = ts
        self.pos = np.array([[px], [py], [pz]])
        self.bbx, self.bby, self.bbz = bbx, bby, bbz
        self.prob = prob
        self.label = label
        self.trk_transform = transform

# Track object
class Track():
    def __init__(self, det, sensor_mdl):
        self.timestamp = det.timestamp
        self.last_updated = det.timestamp
        self.missed_det = 0
        self.prob = det.prob
        self.label = det.label
        self.vel_variance = 0.5

        # Bounding box orientation and size
        self.bbx, self.bby, self.bbz = det.bbx, det.bby, det.bbz
        self.transform = det.trk_transform

        # Kalman filter for this object
        self.kf = gtsam.KalmanFilter(6)
        self.state = self.kf.init(np.vstack((det.pos, np.array([[0.],[0.],[0.]]))), sensor_mdl.obs_cov)

        # Process model matrices
        self.proc_model = np.diag(np.ones(6))
        self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas([1,1,1,1,1,1])
    
    def compute_proc_model(self,dt):
        self.proc_model[0,3], self.proc_model[1,4], self.proc_model[2,5]  = dt, dt, dt

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
        self.state = self.kf.update(self.state, sensor_mdl.obs_model, det.pos, sensor_mdl.obs_noise)
        self.timestamp = det.timestamp
        self.last_updated = det.timestamp
        self.missed_det = 0
        self.bbx, self.bby, self.bbz = det.bbx, det.bby, det.bbz
        self.transform = det.trk_transform

# Graph Tracker object
class Tracker():
    def __init__(self, name, frame, trk_pub, viz_pub):
        # Generic filter states
        self.name = name       
        self.init = False

        # ROS transforms
        self.frame_id = frame
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.pose_stamped = PoseStamped()
        self.sensor_transform = tf2_ros.TransformStamped()

        # Track and detection parameters & variables
        self.detections = []
        self.tracks = []

        # Assignment
        self.asgn_thresh = 4.
        self.cost_matrix = np.empty(0)
        self.det_asgn_idx = [] 
        self.trk_asgn_idx = []

        # Track deletion
        self.trk_delete_thresh = 0.25
        self.trk_delete_time = 3.0
        self.trk_delete_missed_det = 100

        # Initialize member variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.box_msg = BoundingBox()
        self.trk_msg = BoundingBoxArray()
        self.viz_msg = MarkerArray()
        self.trk_pub = trk_pub
        self.viz_pub = viz_pub
    
    def delete_tracks(self):
        self.tracks = [track for track in self.tracks if track.prob > self.trk_delete_thresh]
        self.tracks = [track for track in self.tracks if track.missed_det < self.trk_delete_missed_det]
        self.tracks = [track for track in self.tracks if ((rospy.Time.now() - track.timestamp).to_sec() < self.trk_delete_time)]

    # Data association
    def cost(self, det, track):
        # Euclidean distance between positions
        return np.linalg.norm(det.pos[:,0] - track.state.mean()[0:3])

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
                del self.det_asgn_idx[idx], self.trk_asgn_idx[idx]       
            ii -=1
        assert(len(self.det_asgn_idx) == len(self.trk_asgn_idx))

    def format_trk_msg(self, det_array_msg):
        self.trk_msg = BoundingBoxArray()
        self.trk_msg.header.stamp = det_array_msg.header.stamp
        self.trk_msg.header.frame_id = self.frame_id

        rospy.logdebug("OUTPUT: publishing %i tracks\n\n", len(self.tracks))
        for trk in self.tracks:
            self.box_msg = BoundingBox()
            self.box_msg.header = self.trk_msg.header
            self.box_msg.pose.position.x = trk.state.mean()[0]
            self.box_msg.pose.position.y = trk.state.mean()[1]
            self.box_msg.pose.position.z = trk.state.mean()[2]
            self.box_msg.pose.orientation = trk.transform.transform.rotation
            self.box_msg.dimensions.x, self.box_msg.dimensions.y, self.box_msg.dimensions.z = trk.bbx, trk.bby, trk.bbz
            self.box_msg.value = trk.prob
            self.box_msg.label = trk.label
            self.trk_msg.boxes.append(self.box_msg)


    # Detection callback / main algorithm
    def det_callback(self, det_array_msg, sensor_mdl):
        
        # Populate detections list from detections message
        rospy.logdebug("DETECT: received %i detections", len(det_array_msg.boxes))
        for det in det_array_msg.boxes:
            # Convert to tracker frame and add to detections
            self.pose_stamped = self.tf_buf.transform(PoseStamped(det.header,det.pose), self.frame_id, rospy.Duration(1))
            self.sensor_transform = self.tf_buf.lookup_transform(self.frame_id,det.header.frame_id, det.header.stamp, rospy.Duration(1))
            self.detections.append(Detection(det.header.stamp, self.pose_stamped.pose.position.x, self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.z, det.dimensions.x, det.dimensions.y, det.dimensions.z, det.value, det.label, self.sensor_transform))
        rospy.logdebug("DETECT: formatted %i detections \n", len(self.detections))

        # Propagate existing tracks
        rospy.logdebug("PREDICT: predicting %i tracks\n", len(self.tracks))
        for trk in self.tracks:
            trk.predict(det_array_msg.header.stamp)
        
        # Compute detection-track assignments
        self.compute_cost_matrix()
        self.solve_cost_matrix()
        rospy.logdebug("ASSIGNMENT: cost matrix")
        rospy.logdebug(self.cost_matrix)
        rospy.logdebug("ASSIGNMENT: detection assignments")
        rospy.logdebug(self.det_asgn_idx)
        rospy.logdebug("ASSIGNMENT: track assignments")
        rospy.logdebug(str(self.trk_asgn_idx) + "\n")

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
            if (len(self.detections)-1) in self.det_asgn_idx: # If detection at end of array is matched
                self.detections.pop() # Remove it
            else: self.tracks.append(Track(self.detections.pop(),sensor_mdl)) # Otherwise create a new track

        # Delete tracks as needed
        self.delete_tracks()

        # Convert tracks to track message
        self.format_trk_msg(det_array_msg)

        # Publish tracks
        self.trk_pub.publish(self.trk_msg)
        self.viz_pub.publish(self.viz_msg)

if __name__ == '__main__':
    # Initialize node
    debug = rospy.get_param("tracker_node/debug")
    rospy.init_node("tracker", log_level=rospy.DEBUG) if debug else rospy.init_node("tracker")
    print("Tracker node initialized")

    # Create publishers
    track_pub = rospy.Publisher("tracks_3d", BoundingBoxArray, queue_size=10)
    viz_pub = rospy.Publisher("track_viz", MarkerArray, queue_size=10)

    # Get parameters
    detector_dict = rospy.get_param("tracker")

    # Create detectors according to params
    for name,value in detector_dict.items():
        if value['type']=='graph_tracker':
            # Create tracker object
            tracker = Tracker(name, value['frame'], track_pub, viz_pub)

            # Create detector subscriber
            # TODO (usability) - read in from config file
            oakd_model = SensorModel()
            det_sub = rospy.Subscriber(value['det_topic'], BoundingBoxArray, tracker.det_callback, oakd_model)

    rospy.spin()