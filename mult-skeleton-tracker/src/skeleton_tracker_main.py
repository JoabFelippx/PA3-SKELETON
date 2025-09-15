import os

import cv2
import numpy as np

from video_processor import VideoProcessor 
from fundamental_matrices import FundamentalMatrices
from skeleton_matcher import SkeletonMatcher
from reconstructor_3d import Reconstructor3D
from three_dimentional_tracker import SORT_3D
from visualizer import Visualizer
from utils import get_skeleton_center

# from tracked_person import TrackedPerson
# from skeleton_tracker import SkeletonTracker


def main():
    """
    Função principal que orquestra o pipeline de RECONSTRUÇÃO 3D por frame.
    """
    
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, '..')
    

    config = {

        'calib_path':  f'{base_dir}/calib_cameras',
        'video_path':  f'{base_dir}/videos',
        'yolo_model':  f'{base_dir}/yolov8m-pose.pt',
        'num_cameras': 4,
        'n_keypoints': 17,
        'max_epipolar_dist': 8,
        'min_matching_joints': 5,
        'min_cameras_for_match': 2,
    }


    print("Inicializando módulos...")
    
    video_processor = VideoProcessor(config)
    
    script_dir = os.path.dirname(__file__)
    calib_full_path = os.path.join(script_dir, config['calib_path'])
    
    camera_files = [f"{calib_full_path}/calib_rt{i}.npz" for i in range(1, config['num_cameras'] + 1)]
    geometry = FundamentalMatrices(camera_files)
    projection_matrices = geometry.projection_matrices_all(camera_files)
    fundamentals = geometry.fundamental_matrices_all(camera_files)
    extrinsic_matrices = geometry.get_extrinsic_matrices(camera_files)

    matcher = SkeletonMatcher(fundamentals, config)
    reconstructor = Reconstructor3D(projection_matrices, config) 
    visualizer = Visualizer()
    
    tracker = SORT_3D(max_age=30, min_hits=3, dist_threshold=2.5)
    
    print("Inicialização completa. Iniciando o loop principal...")
    
    while True:
        # Pega o próximo conjunto de frames e anotações
        frames, annotations = video_processor.process_next_frame()
        
        # Se frames for None, significa que os vídeos acabaram
        if frames is None:
            break
        
        
        skeletons_2d, ids_2d = matcher.extract_skeletons_from_annotations(annotations)
        matched_persons = matcher.match(skeletons_2d, ids_2d)
        reconstructed_skeletons = reconstructor.reconstruct_all(matched_persons, annotations)
    
        current_detections_3d = []
        center_to_skeleton_map = {}
        skeletons_por_deteccao = []
        # skeletons_tracker = []
        
        
        for skeleton_data in reconstructed_skeletons:
            if skeleton_data:
                
                hip_center = get_skeleton_center(skeleton_data)
                
                center_point_to_track = None
                
                if hip_center is not None:
                    center_point_to_track = hip_center
                else:
                    points_3d = np.array(list(skeleton_data.values()))
                    if len(points_3d) > 0:
                        center_point_to_track = np.mean(points_3d, axis=0)
                        
                if center_point_to_track is not None:
                    current_detections_3d.append(center_point_to_track)
                    skeletons_por_deteccao.append(skeleton_data)
                
        # for i, skeleton_data in enumerate(reconstructed_skeletons):
        #     if skeleton_data:
        #         points_3d = np.array(list(skeleton_data.values()))
        #         average_point = np.mean(points_3d, axis=0)        
                # skeletons_to_visualize.append({'id': i, 'skeleton_3d': skeleton_data, 'average_point': average_point})
                
        tracker_results = tracker.update(current_detections_3d)
        
        skeletons_to_visualize = []
        tracked_ids = tracker_results['ids']
        tracked_positions = tracker_results['positions']
        
        
        # print('curr', current_detections_3d)
        # print('tracked', tracked_positions)
        
        
        for i in range(len(tracked_ids)):
            track_id = tracked_ids[i]
            filtered_center  = tracked_positions[i]
            
            full_skeleton = None
            
            if len(current_detections_3d) > 0:
                distances = np.linalg.norm(np.array(current_detections_3d) - filtered_center, axis=1)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 0.5:
                    full_skeleton = skeletons_por_deteccao[closest_idx]
                    
            if full_skeleton is not None:
                
                print('track_id', track_id)
                
                skeletons_to_visualize.append({
                'id': track_id,
                'skeleton_3d': full_skeleton, 
                'average_point': filtered_center
            })
            # closest_dist = float('inf')
            # closest_raw_skeleton = None
            # for idx, raw_center in enumerate(current_detections_3d):
            #     dist = np.linalg.norm(filtered_center - raw_center)
            #     if dist < closest_dist:
            #         closest_dist = dist
            #         # Se a distância for muito pequena, consideramos um match
            #         if closest_dist < 0.1: # Use um pequeno limiar de tolerância
            #             closest_raw_skeleton = skeletons_por_deteccao[idx]
                    
            # full_skeleton = closest_raw_skeleton
            # # print(full_skeleton)
            # if full_skeleton is not None:
            #     skeletons_to_visualize.append({
            #     'id': track_id,
            #     'skeleton_3d': full_skeleton, 
            #     'average_point': filtered_center
            # })
            
        # visualizer.update(frames[0], skeletons_to_visualize, extrinsic_matrices)
        # print(f"Pessoas rastreadas: {[p['id'] for p in skeletons_to_visualize]}")
        visualizer.update(frames, skeletons_to_visualize, extrinsic_matrices)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_processor.release()
    cv2.destroyAllWindows()
    print("Processamento finalizado.")

if __name__ == '__main__':
    main()