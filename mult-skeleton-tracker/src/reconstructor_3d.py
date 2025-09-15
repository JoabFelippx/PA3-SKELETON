import numpy as np
from skeleton import Skeleton # Supondo que você tenha a classe Skeleton
from is_msgs.image_pb2 import ObjectAnnotations

class Reconstructor3D:
    """
    Responsável por reconstruir esqueletos 3D a partir de um conjunto
    de esqueletos 2D correspondentes de múltiplas câmeras.
    """
    def __init__(self, projection_matrices, config):
        self.projection_matrices = projection_matrices
        self.config = config

    def _create_skeleton_objects_for_person(self, person_match, all_annotations):
        """Cria os objetos 'Skeleton' para uma pessoa com correspondências encontradas."""
        skeleton_objects = []
        for cam_idx, sk_id in person_match.items():
            if cam_idx < len(all_annotations):
                for sk_obj in all_annotations[cam_idx].objects:
                    if sk_id == sk_obj.id:
                        # Assumindo que Skeleton é uma classe que você tem para encapsular os dados
                        skeleton_objects.append(Skeleton(sk_obj, sk_id, camera_id=cam_idx + 1))
                        break
        return skeleton_objects

    def _to_3d_keypoints_structure(self, skeleton_objs):
        """
        Organiza os keypoints 2D, mapeando cada keypoint para as câmeras que o viram.
        """
        sk_points = np.zeros((self.config['num_cameras'], self.config['n_keypoints'], 2))
        for sk_obj in skeleton_objs:
            cam_idx = sk_obj.camera_id - 1
            for kp in sk_obj.skeleton_obj.keypoints:
                if kp.id < self.config['n_keypoints']:
                    sk_points[cam_idx, kp.id] = [kp.position.x, kp.position.y]
        
        valid_keypoints_info = {}
        for kp_idx in range(self.config['n_keypoints']):
            cameras_with_point = []
            for cam_idx in range(self.config['num_cameras']):
                point = sk_points[cam_idx, kp_idx]
                if not np.allclose(point, [0, 0]):
                    cameras_with_point.append((cam_idx + 1, point))
            if len(cameras_with_point) >= 2: # Precisa de pelo menos 2 vistas para triangular
                valid_keypoints_info[kp_idx] = cameras_with_point
        return valid_keypoints_info

    def _reconstruct_points_from_svd(self, valid_keypoints_info):
        """Reconstrói os pontos 3D usando SVD para os keypoints com correspondências."""
        dots_3d_all = {}
        for kp_idx, cam_data in valid_keypoints_info.items():
            num_cams = len(cam_data)
            A = np.zeros((2 * num_cams, 4)) # Usando o método DLT padrão
            for i, (cam_id, point2d) in enumerate(cam_data):
                P = self.projection_matrices[cam_id - 1]
                A[2*i]   = point2d[0] * P[2,:] - P[0,:]
                A[2*i+1] = point2d[1] * P[2,:] - P[1,:]

            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, 0:4] # Última linha de V (ou última coluna de V transposta)
            
            if X[3] != 0:
                X = X / X[3]
                dots_3d_all[kp_idx] = [X[0], X[1], X[2]]
        return dots_3d_all
    
    def reconstruct_all(self, matched_persons, all_annotations):
        """
        Executa o pipeline de reconstrução completo para todas as pessoas correspondentes.
        """
        reconstructed_skeletons = []
        for person_match in matched_persons:
            skeleton_objects = self._create_skeleton_objects_for_person(person_match, all_annotations)
            valid_kps = self._to_3d_keypoints_structure(skeleton_objects)
            person_3d = self._reconstruct_points_from_svd(valid_kps)
            if person_3d: # Apenas adiciona se a reconstrução foi bem-sucedida
                reconstructed_skeletons.append(person_3d)
        return reconstructed_skeletons