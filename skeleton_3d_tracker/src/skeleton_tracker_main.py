# skeleton_tracker_main.py

import time
import cv2
import numpy as np

from stream_handler import StreamHandler
from fundamental_matrices import FundamentalMatrices
from skeleton_matcher import SkeletonMatcher
from reconstructor_3d import Reconstructor3D
from three_dimentional_tracker import SORT_3D
from visualizer import Visualizer
from utils import get_skeleton_center

def main():
    config = {
        'broker_uri': "amqp://guest:guest@10.10.2.211:30000",
        'calib_path': '../calib_cameras',
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
    stream = StreamHandler(config)
    
    camera_files = [f"{config['calib_path']}/calib_rt{i}.npz" for i in range(1, config['num_cameras'] + 1)]
    calib_files_data = [np.load(f) for f in camera_files]
    
    geometry = FundamentalMatrices()
    projection_matrices = geometry.projection_matrices_all(camera_files)
    fundamentals = geometry.fundamental_matrices_all(camera_files)
    extrinsic_matrices = geometry.get_extrinsic_matrices(camera_files)

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
        raw_messages = stream.get_latest_messages()
        if raw_messages is None:
            time.sleep(0.01)
            continue
        
        images, annotations = stream.prepare_input_data(raw_messages, calib_files_data)
        
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
        # MUDANÇA: Cria uma lista para guardar as trajetórias associadas
        tracked_trajectories = []
        
        if tracked_data['positions']:
            for i, tracked_center in enumerate(tracked_data['positions']):
                track_id = tracked_data['ids'][i]
                
                distances = [np.linalg.norm(tracked_center - item['center']) for item in center_to_skeleton_map]
                if distances:
                    best_match_idx = np.argmin(distances)
                    matched_skeleton = center_to_skeleton_map[best_match_idx]['skeleton']
                    tracked_skeletons_with_ids.append({'id': track_id, 'skeleton': matched_skeleton})
                    # MUDANÇA: Associa a trajetória ao esqueleto
                    tracked_trajectories.append({'id': track_id, 'trajectory': tracked_data['trajectories'][i]})

        # MUDANÇA: Passa a lista de trajetórias para o visualizador
        visualizer.update(images[0], tracked_skeletons_with_ids, extrinsic_matrices, tracked_trajectories)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Processamento finalizado.")

if __name__ == '__main__':
    main()