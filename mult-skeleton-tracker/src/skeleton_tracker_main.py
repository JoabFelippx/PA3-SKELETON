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

def main():
    
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, '..')
    
    config = {

        'calib_path': os.path.join(base_dir, 'calib_cameras'),
        'video_path': os.path.join(base_dir, 'videos'),
        'yolo_model': os.path.join(base_dir, 'yolov8m-pose.pt'),
        'num_cameras': 4,
        'n_keypoints': 17,
        'max_epipolar_dist': 8,
        'min_matching_joints': 5,
        'min_cameras_for_match': 2,
        'tracker_max_age': 60,
        'tracker_min_hits': 3,
        'tracker_dist_threshold': 1.5
    }

    print("Inicializando módulos...")
    
    video_processor = VideoProcessor(config)
    
    script_dir = os.path.dirname(__file__)
    calib_full_path = os.path.join(script_dir, '..', config['calib_path'])
    
    camera_files = [f"{calib_full_path}/calib_rt{i}.npz" for i in range(1, config['num_cameras'] + 1)]
    geometry = FundamentalMatrices(camera_files)
    projection_matrices = geometry.projection_matrices_all()
    fundamentals = geometry.fundamental_matrices_all()
    extrinsic_matrices = geometry.get_extrinsic_matrices()

    matcher = SkeletonMatcher(fundamentals, config)
    reconstructor = Reconstructor3D(projection_matrices, config)
    tracker = SORT_3D(
        max_age=config['tracker_max_age'], 
        min_hits=config['tracker_min_hits'], 
        dist_threshold=config['tracker_dist_threshold']
    )
    visualizer = Visualizer()
    print("Inicialização completa. Iniciando o loop principal...")
    
    while True:
        # Pega o próximo conjunto de frames e anotações
        frames, annotations = video_processor.process_next_frame()
        
        # Se frames for None, significa que os vídeos acabaram
        if frames is None:
            break
        
        # O resto do pipeline continua exatamente o mesmo!
        skeletons_2d, ids_2d = matcher.extract_skeletons_from_annotations(annotations)
        matched_persons = matcher.match(skeletons_2d, ids_2d)
        reconstructed_skeletons = reconstructor.reconstruct_all(matched_persons, annotations)

        skeleton_centers = []
        center_to_skeleton_map = [] 
        for skel_3d in reconstructed_skeletons:
            center = get_skeleton_center(skel_3d)
            if center is not None:
                skeleton_centers.append(center)
                center_to_skeleton_map.append({'center': center, 'skeleton': skel_3d})

        if len(skeleton_centers) > 0:
            tracked_data = tracker.update(np.array(skeleton_centers))
        else:
            tracked_data = tracker.update(np.empty((0, 3)))
        
        tracked_skeletons_with_ids = []
        tracked_trajectories = []
        if tracked_data['positions']:
            for i, tracked_center in enumerate(tracked_data['positions']):
                track_id = tracked_data['ids'][i]
                distances = [np.linalg.norm(tracked_center - item['center']) for item in center_to_skeleton_map]
                if distances:
                    best_match_idx = np.argmin(distances)
                    matched_skeleton = center_to_skeleton_map[best_match_idx]['skeleton']
                    tracked_skeletons_with_ids.append({'id': track_id, 'skeleton': matched_skeleton})
                    if 'trajectories' in tracked_data:
                         tracked_trajectories.append({'id': track_id, 'trajectory': tracked_data['trajectories'][i]})

        visualizer.update(frames[0], tracked_skeletons_with_ids, extrinsic_matrices, tracked_trajectories)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_processor.release()
    cv2.destroyAllWindows()
    print("Processamento finalizado.")

if __name__ == '__main__':
    main()