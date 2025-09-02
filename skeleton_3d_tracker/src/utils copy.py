# utils.py
import numpy as np

def get_skeleton_center(skeleton_3d: dict, hip_indices: list = [11, 12]) -> np.ndarray or None:
    """
    Calcula o ponto central de um esqueleto 3D (média dos quadris).
    
    Args:
        skeleton_3d (dict): Dicionário {kp_id: [x, y, z]} representando um esqueleto.
        hip_indices (list): Lista dos IDs dos keypoints a serem usados para o centro.
    
    Returns:
        np.ndarray or None: As coordenadas [x, y, z] do ponto central, ou None se não for possível calcular.
    """
    # Pega os pontos do quadril esquerdo e direito que existem no dicionário
    hip_points = [skeleton_3d.get(idx) for idx in hip_indices if skeleton_3d.get(idx) is not None]
    
    # Fallback: se os quadris não estiverem visíveis, usa a média de todos os pontos disponíveis
    if not hip_points:
        if not skeleton_3d:
            return None
        hip_points = list(skeleton_3d.values())

    # Calcula a média dos pontos para encontrar o centro
    center_point = np.mean(np.array(hip_points), axis=0)
    return center_point