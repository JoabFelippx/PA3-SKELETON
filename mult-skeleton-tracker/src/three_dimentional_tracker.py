"""
SORT 3D: Simple Online and Realtime Tracking in 3D

This module implements 3D tracking using the SORT algorithm adapted for 3D space.
The original SORT algorithm is extended to work with 3D positions instead of 2D bounding boxes.
It uses Kalman filtering to predict and update object positions in 3D space.
`
Reference:
SORT: A Simple, Online and Realtime Tracker
https://arxiv.org/abs/1602.00763
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def linear_interpolate(start_point, end_point, num_steps):
    
    """
        num_steps = quantos frames o obj ficou sem ser visto
        vai ser o numero de points gerados para criar a trajetoria virtual
    """
    if num_steps <= 0: 
        return []
    
    intermediate_points = []
    displacement_vector = end_point - start_point

    for i in range(1, num_steps + 1):
            fractional_step = i / (num_steps + 1.0)
            point = start_point + fractional_step * displacement_vector
            intermediate_points.append(point)
    
    
    print('intermediate_points', intermediate_points)
    return intermediate_points


class KalmanSktCenterTracker:
    """
    This class represents the internal state of individual tracked objects observed as 3D points.
    The state includes position and velocity in 3D space.
    """
    count = 0

    def __init__(self, point_3d, class_id=None):
        """
        Initialize a tracker using a 3D point.
        
        Args:
            point_3d (numpy.ndarray): 3D position [x, y, z]
            class_id (int, optional): Class ID of the object
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 1, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 1],  # z = z + dz
            [0, 0, 0, 1, 0, 0],  # dx = dx
            [0, 0, 0, 0, 1, 0],  # dy = dy
            [0, 0, 0, 0, 0, 1]   # dz = dz
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement uncertainty
        self.kf.R = np.eye(3) * 0.1
        
        # Process uncertainty
        self.kf.Q = np.eye(6) * 0.1
        self.kf.Q[3:, 3:] *= 0.5  # Higher uncertainty for velocity
        
        # Initial state uncertainty
        self.kf.P = np.eye(6) * 10
        
        # Initialize state
        self.kf.x = np.zeros(6)
        self.kf.x[:3] = point_3d
        
        self.last_good_x = self.kf.x.copy()
        self.last_good_P = self.kf.P.copy()
        
        # Assign ID and initialize counters
        self.id = KalmanSktCenterTracker.count
        KalmanSktCenterTracker.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.trajectory = [point_3d]  # Store trajectory for visualization
        
        self.observations = [point_3d]
        
        # Store class information
        self.class_id = class_id
        
        
    def reupdate_with_virtual_trajectory(self, virtual_trajectory, final_real_observation):

        
        self.kf.x = self.last_good_x.copy()
        self.kf.P = self.last_good_P.copy()
        
        for virtual_point in virtual_trajectory:
            self.kf.predict()
            self.kf.update(virtual_point)
        
        self.kf.predict()
        self.kf.update(final_real_observation)
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.trajectory.append(self.get_position())
        
        self.observations.append(final_real_observation)
        self.last_good_x = self.kf.x.copy()
        self.last_good_P = self.kf.P.copy()

    def update(self, point_3d, class_id=None):
        """
        Update the state using a new 3D point.
        
        Args:
            point_3d (numpy.ndarray): 3D position [x, y, z]
            class_id (int, optional): Class ID of the object
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point_3d)
        self.trajectory.append(self.get_position())
        
        self.last_good_x = self.kf.x.copy()
        self.last_good_P = self.kf.P.copy()
        
        
        self.observations.append(point_3d)
        
        # Update class if provided (for consistency checking)
        if class_id is not None and self.class_id != class_id:
            # Log class change if it happens (might indicate tracking error)
            if self.class_id is not None:
                print(f"Warning: Class changed for track {self.id}: {self.class_id} -> {class_id}")
            self.class_id = class_id

    def predict(self):
        """
        Advances the state vector and returns the predicted position.
        
        Returns:
            numpy.ndarray: Predicted 3D position [x, y, z]
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 0 else self.hit_streak
        return self.kf.x[:3]
        
    def get_position(self):
        """
        Returns the current position in 3D space.
        
        Returns:
            numpy.ndarray: Current 3D position [x, y, z]
        """
        return self.kf.x[:3].copy()
    
    def get_state(self):
        """
        Returns the current state (position and velocity).
        
        Returns:
            numpy.ndarray: Current state [x, y, z, dx, dy, dz]
        """
        return self.kf.x.copy()


class SORT_3D:
    """
    SORT algorithm adapted for 3D tracking.
    """
    
    def __init__(self, max_age=60, min_hits=3, dist_threshold=2.5):
        """
        Initialize the SORT 3D tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep a track without matching detection
            min_hits (int): Minimum number of matched detections before track is initialized
            dist_threshold (float): Maximum distance for associating detections with tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0
    
    # def _calculate_ocm_cost(self,track, detection_point, delta_t=3):
    #     if len(track.observations) < delta_t:
    #         return 0.0 
        
    #     last_observation = track.observations[-1]
    #     past_observations = track.observations[-delta_t]
        
    #     v_track = last_observation - past_observations # vetor de direção com o que ja tinha 
     
    #     norm_v_track = np.linalg.norm(v_track) # normalizar para evitar problemas com magnitude
    #     if norm_v_track > 0:
    #         v_track /= norm_v_track
            
    #     v_intention = detection_point - last_observation # vetor com a nova detecção 
        
    #     norm_v_intention = np.linalg.norm(v_intention)
    #     if norm_v_intention > 0:
    #         v_intention /= norm_v_intention
        
    #     # nesse caso 1 = bom e -1 = ruim
    #     dot_product = np.dot(v_track, v_intention)
        
    #     dot_product = np.clip(dot_product, -1.0, 1.0) 
        
    #     angle_cost = np.arccos(dot_product) # angulo entre os vetores v_track e v_intention
    
    
    #     print("angle_cost:", angle_cost)
    #     return angle_cost
        
        
    def _calculate_ocm_cost(self, track, detection_point, delta_t=3):
    
        if len(track.observations) < 2:
            return 0.0

        recent_speed = np.linalg.norm(track.observations[-1] - track.observations[-2])
        
        speed_threshold = 0.045 
        
        
        print('recent_speed', recent_speed)
        if recent_speed < speed_threshold:
            return 0.0 

        if len(track.observations) < delta_t:
            return 0.0 
        
        last_observation = track.observations[-1]
      
        past_observation = track.observations[-delta_t] 
        
        v_track = last_observation - past_observation
    
        norm_v_track = np.linalg.norm(v_track)
        if norm_v_track > 0:
            v_track /= norm_v_track
            
        v_intention = detection_point - last_observation
        
        norm_v_intention = np.linalg.norm(v_intention)
        if norm_v_intention > 0:
            v_intention /= norm_v_intention
        
        dot_product = np.dot(v_track, v_intention)
        dot_product = np.clip(dot_product, -1.0, 1.0) 
        angle_cost = np.arccos(dot_product)

        return angle_cost
    
    def update(self, points_3d, class_ids=None):
        """
        Updates the tracker with new 3D detections.
        
        Args:
            points_3d (list or numpy.ndarray): List of 3D points detected in current frame
                                               Each point is [x, y, z]
            class_ids (list, optional): List of class IDs corresponding to each point
        
        Returns:
            dict: Dictionary containing:
                - 'positions': list of filtered 3D positions
                - 'ids': list of corresponding track IDs
                - 'class_ids': list of corresponding class IDs
                - 'trajectories': dictionary mapping track IDs to trajectories
        """
        self.frame_count += 1
        
        lambda_weight = 0.2
        
        # Convert input to numpy array if it isn't already
        points_3d = np.array(points_3d) if len(points_3d) > 0 else np.empty((0, 3))
        class_ids = class_ids if class_ids is not None else [None] * len(points_3d)
        
        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Filter out trackers with NaN predictions
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers using distance matrix and Hungarian algorithm
        
        lambda_weight = 0.6
        if len(trks) > 0 and len(points_3d) > 0:
            # Compute distance matrix between predictions and detections
            dist_matrix = np.zeros((len(points_3d), len(trks)))
            for i in range(len(points_3d)):
                for j in range(len(trks)):
                    # Euclidean distance in 3D
                    cost_pos = np.linalg.norm(points_3d[i] - trks[j])
                    cost_ocm = self._calculate_ocm_cost(self.trackers[j], points_3d[i])
                    
                    print("cost_pos", cost_pos)
                    print("cost_ocm", cost_ocm)
                    
                    
                    dist_matrix[i, j] = cost_pos + lambda_weight * cost_ocm 
            # Use Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(dist_matrix)
            
            # Only keep assignments below the distance threshold
            valid_matches = dist_matrix[row_indices, col_indices] < self.dist_threshold
            print('valid_matches', dist_matrix[row_indices, col_indices])
            
            # Create assignment lists
            matches = []
            unmatched_detections = list(range(len(points_3d)))
            unmatched_trackers = list(range(len(trks)))
            
            for row, col, valid in zip(row_indices, col_indices, valid_matches):
                if valid:
                    matches.append((row, col))
                    unmatched_detections.remove(row)
                    unmatched_trackers.remove(col)
        else:
            matches = []
            unmatched_detections = list(range(len(points_3d)))
            unmatched_trackers = list(range(len(trks)))
            
        # Update matched trackers
        for det_idx, trk_idx in matches:
            trk = self.trackers[trk_idx]
            
            
            if trk.time_since_update > 0: # caso de oclusão 
                last_obs = trk.observations[-1]
                new_obs = points_3d[det_idx]
                lost_frames = trk.time_since_update
                
                virtual_trajectory = linear_interpolate(last_obs, new_obs, lost_frames)
                
                trk.reupdate_with_virtual_trajectory(virtual_trajectory, new_obs)
            else:
                trk.update(points_3d[det_idx], class_ids[det_idx])

        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            new_tracker = KalmanSktCenterTracker(points_3d[det_idx], class_ids[det_idx])
            self.trackers.append(new_tracker)
            
        # Remove dead tracklets
        i = len(self.trackers) - 1
        while i >= 0:
            trk = self.trackers[i]
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            i -= 1
            
        # Prepare output
        ret_positions = []
        ret_ids = []
        ret_class_ids = []
        ret_trajectories = {}
        
        for trk in self.trackers:
            # Only return tracks that have been confirmed (hit min_hits)
            if trk.hits >= self.min_hits and trk.time_since_update <= 1:
                position = trk.get_position()
                ret_positions.append(position)
                ret_ids.append(trk.id)
                ret_class_ids.append(trk.class_id)
                ret_trajectories[trk.id] = trk.trajectory
                
        return {
            'positions': ret_positions,
            'ids': ret_ids,
            'class_ids': ret_class_ids,
            'trajectories': ret_trajectories
        }

# Testing code
if __name__ == "__main__":
    # Simple test to verify the SORT 3D implementation
    sort_tracker = SORT_3D(max_age=5, min_hits=2, dist_threshold=1.0)
    
    # Simulate points moving in 3D
    all_points = [
        [[1, 1, 1], [4, 4, 0]],
        [[1.1, 1.1, 1.05], [4.1, 4.1, 0.02]],
        [[1.2, 1.15, 1.1], [4.15, 4.2, 0.04]],
        [[1.3, 1.2, 1.15]],
        [[1.35, 1.25, 1.2]],
    ]
    
    for frame, points in enumerate(all_points):
        print(f"Frame {frame}:")
        result = sort_tracker.update(points)
        print(f"  Positions: {result['positions']}")
        print(f"  IDs: {result['ids']}")