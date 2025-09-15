import numpy as np
import networkx as nx
from is_msgs.image_pb2 import ObjectAnnotations

class SkeletonMatcher:
    
    def __init__(self, fundamentals: dict, config: dict):
        
        self.fundamentals = fundamentals
        self.config = config
    
    def _dist_p_l(self, point, line):
        
        x, y = point
        a, b, c = line
        
        denominator = np.sqrt(a**2 + b**2) 
        if denominator == 0:
            return float('inf')
        
        return abs(a * x + b * y + c) / denominator
    
    def _calculate_skeleton_compatibility(self, sk1, sk2, F_1_to_2):
        
        score = 0
        for kp1, kp2, in zip(sk1, sk2):
            
            if np.allclose(kp1, [0, 0]) or np.allclose(kp2, [0, 0]):
                continue
            
            epiline_on_2 = F_1_to_2 @ np.array([kp1[0], kp1[1], 1])
            distance = self._dist_p_l(kp2, epiline_on_2)
            
            if distance < self.config['max_epipolar_dist']:
                score += 1 
        
        return score
    
    def extract_skeletons_from_annotations(self, annotations):
        
        skeletons_by_cam = []
        ids_by_cam = []

        for i in range(self.config['num_cameras']):
            
            skeletons_for_current_cam = []
            ids_for_current_cam = []

            if i < len(annotations):
                for obj in annotations[i].objects:
                    skeleton = np.zeros((self.config['n_keypoints'], 2))
                    for kp in obj.keypoints:
                        if kp.id < self.config['n_keypoints']:
                            skeleton[kp.id] = [kp.position.x, kp.position.y]
                    
                    skeletons_for_current_cam.append(skeleton)
                    ids_for_current_cam.append(obj.id)
            
            skeletons_by_cam.append(skeletons_for_current_cam)
            ids_by_cam.append(ids_for_current_cam)
            
        return skeletons_by_cam, ids_by_cam
    
    def match(self, skeletons_by_cam: list, ids_by_cam: list):

        all_pairs = []
        
        for cam_idx1 in range(self.config['num_cameras']):
            for cam_idx2 in range(cam_idx1 + 1, self.config['num_cameras']):
                
                skeletons1 = skeletons_by_cam[cam_idx1]
                ids1 = ids_by_cam[cam_idx1]
                
                skeletons2 = skeletons_by_cam[cam_idx2]
                ids2 = ids_by_cam[cam_idx2]
                
                F_1_to_2 = self.fundamentals.get(cam_idx1, {}).get(cam_idx2)
                if F_1_to_2 is None:
                    continue

                for i1, sk1 in enumerate(skeletons1):
                    best_score = -1
                    best_match_idx = -1
                    
                    for i2, sk2 in enumerate(skeletons2):
                        score = self._calculate_skeleton_compatibility(sk1, sk2, F_1_to_2)
                        if score > best_score:
                            best_score = score
                            best_match_idx = i2
                    
                    if best_score >= self.config['min_matching_joints']:
                        pair = ( (cam_idx1, ids1[i1]), (cam_idx2, ids2[best_match_idx]) )
                        all_pairs.append(pair)

        
        G = nx.Graph()
        for pair in all_pairs:
            G.add_edge(pair[0], pair[1])
        
       
        connected_components = list(nx.connected_components(G))
        
        matched_persons = []
        for component in connected_components:
            if len(component) < 2:
                continue
            
            person_match = {}
            for detection in component:
                cam_idx, person_id = detection
                person_match[cam_idx] = person_id
            
            matched_persons.append(person_match)
            
        return matched_persons
    
    # --- CAMERA 1 COMO REF
    # def match(self, skeletons_by_cam, ids_by_cam):
        
    #     matched_persons = []
    #     used_ids = {i: set() for i in range(self.config)}
        
    #     skeletons_cam_ref = skeletons_by_cam[0]
    #     ids_cam_ref = ids_by_cam[0]
        
    #     for i_ref, sk_ref in enumerate(skeletons_cam_ref):
    #         if ids_cam_ref[i_ref] in used_ids[0]:
    #             continue
            
    #         best_matches = {}
            
    #         for cam_idx in range(1, self.config['num_cameras']):
    #             best_score = -1
    #             best_cand_idx = -1
                
    #             F_ref_to_N = self.fundamentals.get(0, {}).get(cam_idx)
                
    #             if F_ref_to_N is None: 
    #                 continue

    #             for i_cand, sk_cand in enumerate(skeletons_by_cam[cam_idx]):
    #                 if ids_by_cam[cam_idx][i_cand] in used_ids[cam_idx]:
    #                     continue
                    
    #                 score = self._calculate_skeleton_compatibility(sk_ref, sk_cand, F_ref_to_N)
                    
    #                 if score > best_score:
    #                     best_score = score
    #                     best_cand_idx = i_cand
                
    #             if best_score >= self.config['min_matching_joints']:
    #                 best_matches[cam_idx] = {'id_idx': best_cand_idx, 'score': best_score}
            
    #         if len(best_matches) >= self.config.get('min_cameras_for_match', 1):
    #             person_match = {0: ids_cam_ref[i_ref]}
    #             for cam_idx, match_info in best_matches.items():
    #                  person_match[cam_idx] = ids_by_cam[cam_idx][match_info['id_idx']]
                
    #             matched_persons.append(person_match)

    #             for cam_idx, sk_id in person_match.items():
    #                 used_ids[cam_idx].add(sk_id)

    #     return matched_persons
                