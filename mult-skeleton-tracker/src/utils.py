# utils.py

import cv2
import numpy as np

# ============================================================================
# Funções Principais para o Pipeline de Rastreamento 3D
# ============================================================================

def get_skeleton_center(skeleton_3d: dict, hip_indices: list = [11, 12]) -> np.ndarray or None:
    """
    Calcula o ponto central de um esqueleto 3D (média dos quadris).
    Retorna None se os pontos do quadril não estiverem disponíveis.
    """
    # Pega os pontos do quadril esquerdo e direito que existem no dicionário
    hip_points = [skeleton_3d.get(idx) for idx in hip_indices if skeleton_3d.get(idx) is not None]
    
    # Se não houver pontos de quadril suficientes (precisamos de pelo menos um),
    # a função agora retorna None.
    if len(hip_points) < 1:
        return None

    # Calcula a média dos pontos do quadril para encontrar o centro.
    center_point = np.mean(np.array(hip_points), axis=0)
    return center_point

# ============================================================================
# Funções Auxiliares de Processamento de Imagem (do seu código)
# ============================================================================

def calib_img_from_file(npzCalib, image):
    """Aplica a calibração para remover a distorção da imagem."""
    undistort_img = cv2.undistort(image, npzCalib['K'], npzCalib['dist'], None, npzCalib['nK'])
    if 'roi' in npzCalib:
        x, y, w, h = npzCalib['roi']
        return undistort_img[y:y+h, x:x+w]
    return undistort_img

# ============================================================================
# Funções de Desenho 2D (Úteis para Depuração)
# O pipeline principal não as utiliza, mas você pode usá-las para
# visualizar as detecções 2D antes da reconstrução 3D.
# ============================================================================

def draw_bounding_box(image, bbox_xyxy, color=(150, 0, 0), thickness=2):
    """Desenha uma caixa delimitadora na imagem."""
    return cv2.rectangle(image, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])), color, thickness)

def draw_identifier(image, bbox_xyxy, person_id, color=(150, 0, 0)):
    """Desenha o ID da pessoa na imagem."""
    offset_x = 0
    offset_y = -10
    font_size = 1
    font_thickness = 2
    
    position = (int(bbox_xyxy[0] + offset_x), int(bbox_xyxy[1] + offset_y))
    
    return cv2.putText(image, str(person_id), position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)

def draw_skeleton(image, keypoints, skeleton_map):
    """Desenha as linhas (ossos) do esqueleto na imagem."""
    for connection in skeleton_map:
        srt_kpt_id = connection['srt_kpt_id']
        dst_kpt_id = connection['dst_kpt_id']
        
        # Pega os pontos de início e fim
        p1 = keypoints[srt_kpt_id]
        p2 = keypoints[dst_kpt_id]

        # Verifica se ambos os pontos foram detectados
        if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
            continue

        color = connection.get('color', (0, 255, 0))
        thickness = connection.get('thickness', 2)

        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    return image

def draw_keypoints(image, keypoints, kpt_color_map):
    """Desenha os círculos (articulações) do esqueleto na imagem."""
    for kpt_id, data in kpt_color_map.items():
        point = keypoints[kpt_id]
        
        # Verifica se o ponto foi detectado
        if point[0] == 0 and point[1] == 0:
            continue

        color = data.get('color', (0, 0, 255))
        radius = data.get('radius', 4)

        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    return image