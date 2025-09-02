# three_dimentional_tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    count = 0
    def __init__(self, point_3d):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([[1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
        self.kf.R *= 10.
        self.kf.P[3:,3:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[3:,3:] *= 0.01
        self.kf.Q = self.kf.Q * 0.1
        self.kf.x[:3] = point_3d.reshape((3, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # MUDANÇA: Inicializa o histórico com a primeira posição
        self.history = [self.get_state().flatten()]
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, point_3d):
        self.time_since_update = 0
        # MUDANÇA: Adiciona a nova posição ao histórico
        self.history.append(self.get_state().flatten())
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point_3d.reshape((3, 1)))

    def predict(self):
        if((self.kf.x[3]+self.kf.x[4]) <= 0.):
            self.kf.x[3] *= 0.0
            self.kf.x[4] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:3].reshape((1, 3))

    def get_state(self):
        return self.kf.x[:3].reshape((1, 3))

def associate_detections_to_trackers(detections, trackers, distance_threshold = 0.3):
    if(len(trackers) == 0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            distance_matrix[d,t] = np.linalg.norm(det - trk)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))
    if matched_indices.size > 0:
        matches_below_threshold = []
        for m in matched_indices:
            if distance_matrix[m[0], m[1]] <= distance_threshold:
                matches_below_threshold.append(m.reshape(1, 2))
        if len(matches_below_threshold) > 0:
            matches = np.concatenate(matches_below_threshold, axis=0)
            unmatched_detections = np.array([d for d in range(len(detections)) if d not in matches[:, 0]])
            unmatched_trackers = np.array([t for t in range(len(trackers)) if t not in matches[:, 1]])
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_detections = np.arange(len(detections))
            unmatched_trackers = np.arange(len(trackers))
    else:
        matches = np.empty((0, 2), dtype=int)
        unmatched_detections = np.arange(len(detections))
        unmatched_trackers = np.arange(len(trackers))
    return matches, unmatched_detections, unmatched_trackers

class SORT_3D:
    def __init__(self, max_age=10, min_hits=3, dist_threshold=0.8):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0
    def update(self, points_3d: np.ndarray):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trks[t,:] = [pos[0],pos[1],pos[2]]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if points_3d.ndim == 1 and points_3d.size > 0:
            points_3d = points_3d.reshape(1, -1)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(points_3d, trks, self.dist_threshold)
        for m in matched:
            self.trackers[m[1]].update(points_3d[m[0],:])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(points_3d[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        # MUDANÇA: Adiciona 'trajectories' ao dicionário de retorno
        ret = {'positions': [], 'ids': [], 'trajectories': []}
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret['positions'].append(d)
                ret['ids'].append(trk.id)
                # MUDANÇA: Adiciona o histórico do rastreador
                ret['trajectories'].append(trk.history)
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        return ret