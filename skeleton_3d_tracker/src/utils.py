# utils.py
import numpy as np

def get_skeleton_center(skeleton_3d: dict, hip_indices: list = [11, 12]) -> np.ndarray or None:
    """
    Calcula o ponto central de um esqueleto 3D (média dos quadris).
    Retorna None se os pontos do quadril não estiverem disponíveis.
    """
    # Pega os pontos do quadril esquerdo e direito que existem no dicionário
    hip_points = [skeleton_3d.get(idx) for idx in hip_indices if skeleton_3d.get(idx) is not None]
    
    # --- MUDANÇA AQUI ---
    # Se não houver pontos de quadril suficientes (precisamos de pelo menos um),
    # a função agora retorna None. Não há mais o fallback para a média de todos os pontos.
    if len(hip_points) < 1:
        return None

    # Calcula a média dos pontos do quadril para encontrar o centro.
    center_point = np.mean(np.array(hip_points), axis=0)
    return center_point