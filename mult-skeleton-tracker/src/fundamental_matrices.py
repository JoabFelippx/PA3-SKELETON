
import numpy as np

class FundamentalMatrices:
    
    def __init__(self, camera_files: list):
        self.camera_files = camera_files # Armazena a lista de arquivos

    def get_extrinsic_matrices(self, camera_files):
        """
        Retorna a lista de matrizes extrínsecas [R|t] (3x4) para todas as câmeras.
        """
        # A informação já foi carregada no __init__, apenas precisamos retorná-la
        extrinsic_list = []
        for c in range(len(camera_files)): # Supondo que você armazene camera_files no __init__
            rt, _, _, _= self._load_camera_parameters(camera_files[c])
            extrinsic_list.append(rt)
        return extrinsic_list


    def _load_camera_parameters(self, calibration: str):
        
        camera_data = np.load(calibration)
     
        # Intrinsic matrix (3x3)
        nK = camera_data['nK']
        
        # Image resolution (width, height)
        # res = [camera_data['w'], camera_data['h']]

        # Matrix (4x4) containing rotation and translation      
        rt = camera_data['rt']
        
        # Rotation matrix (3x3)
        R = rt[:3, :3]
        
        # Translation vector
        T = rt[:3, 3].reshape(3, 1)
        
        return rt, R, T, nK
    
    def _calculate_fundamental_matrix(self, K1: np.ndarray, K2: np.ndarray, RT1: np.ndarray, RT2: np.ndarray):
        
        """
        Computes the fundamental matrix between two cameras.

        Args:
            K1 (np.ndarray): Undistorted intrinsic matrix of the first camera.
            K2 (np.ndarray): Undistorted intrinsic matrix of the second camera.
            RT1 (np.ndarray): Transformation matrix (4x4) of the first camera.
            RT2 (np.ndarray): Transformation matrix (4x4) of the second camera.

        Returns:
            np.ndarray: The fundamental matrix (3x3) between the two cameras.
        """
        # Extrinsic matrices (rt) from 3x4 to 4x4
        RT1_4x4 = np.vstack([RT1, [0, 0, 0, 1]])
        RT2_4x4 = np.vstack([RT2, [0, 0, 0, 1]])
        
        # Compute the Relative transformation matrix between the two cameras
        RT_2_1 = RT2_4x4 @ np.linalg.inv(RT1_4x4)
        
        # Extract the rotation and translation between the two cameras
        R_rel = RT_2_1[0:3, 0:3]
        T_rel = RT_2_1[0:3, 3] 
        
        # Compute the skew-symmetric matrix transformation matrix between the two cameras
        skew_symm = np.array([[0, -T_rel[2], T_rel[1]], [T_rel[2], 0, -T_rel[0]], [-T_rel[1], T_rel[0], 0]])
        
        # Compute the essential matrix: E = skew_symm * R_rel
        essential_matrix =  skew_symm @ R_rel
        
        # Compute the fundamental matrix: F = inv(K2)^T * E * inv(K1)
        F = (np.linalg.inv(K2).T) @ essential_matrix @ (np.linalg.inv(K1))
        
        return F
    
    def _calculate_projection_matrix(self, K: np.ndarray, T: np.ndarray):
        """
        Computes the projection matrix for a camera given its intrinsic and extrinsic parameters.

        Args:
            K (np.ndarray): Intrinsic matrix of the camera.
            T (np.ndarray): Translation vector of the camera.

        Returns:
            np.ndarray: The projection matrix (3x4) of the camera.
        """
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        return K @ T
    
    def projection_matrices_all(self, camera_files: list):
        
        """
        Calculates the projection matrices for all cameras.

        Args:
            camera_files (list): A list of file paths to camera calibration files (NPZ format).

        Returns:
            dict: A dictionary where the keys are camera indices, and the values
            are the corresponding projection matrices for each camera.
        """
        
        # Dictionary to store the projection matrices for all cameras
        P_all_cameras = dict()
        
        # Loop over all cameras
        for c in range(len(camera_files)):
            
            # Load the camera parameters for the camera (c)
            
            
            rt, R, T, nK = self._load_camera_parameters(camera_files[c])
            
            # Calculate the projection matrix for the camera
            P = self._calculate_projection_matrix(nK, rt)

            # Store the projection matrix in the dictionary
            P_all_cameras[c] = P
        
        return P_all_cameras
    
    def fundamental_matrices_all(self, camera_files: list):
        """
        Calculates the fundamental matrices between all pairs of cameras.

        Args:
            camera_files (list): A list of file paths to camera calibration files (NPZ format).

        Returns:
            dict: A nested dictionary where the keys are camera indices, and the values
            are the corresponding fundamental matrices between each pair of cameras.
        """
        
        # Dictionary to store the fundamental matrices for all camera pairs
        F_all_camera_pairs = dict()
        
        for c_s in range(len(camera_files)):
            for c_d in range(len(camera_files)):
                if c_s == c_d:  # Skip pairs where the source and destination cameras are the same
                    continue
                
                # Load the camera parameters for the source camera (C_s) and destination camera (C_d)
                rt1, R1, T1, nK1 = self._load_camera_parameters(camera_files[c_s])
                rt2, R2, T2, nK2 = self._load_camera_parameters(camera_files[c_d])
                
                 # Calculate the fundamental matrix between the source and destination cameras
                F = self._calculate_fundamental_matrix(nK1, nK2, rt1, rt2)
                
                # Store the fundamental matrix in a nested dictionary
                if c_s not in F_all_camera_pairs:
                    F_all_camera_pairs[c_s] = {}
                F_all_camera_pairs[c_s][c_d] = F
                
        return F_all_camera_pairs
        