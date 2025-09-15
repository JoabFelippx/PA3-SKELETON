import cv2
import numpy as np
from skeletons import SkeletonsDetector
from is_msgs.image_pb2 import ObjectAnnotations 

import os

class VideoProcessor:
    """
    Responsável por abrir múltiplos vídeos, ler os frames em sincronia,
    executar a detecção de esqueletos e fornecer os dados para o pipeline de rastreamento.
    """
    def __init__(self, config: dict):
        self.config = config
        self.num_cameras = config['num_cameras']
        
        print(f"Carregando modelo YOLO: {config['yolo_model']}")
        detector_options = {'model': config['yolo_model']}
        self.detector = SkeletonsDetector(detector_options)
        
        self.calib_data = [np.load(f"{config['calib_path']}/calib_rt{i}.npz") for i in range(1, self.num_cameras + 1)]

        self.video_captures = []
        video_base_path = config['video_path']
        for i in range(1, self.num_cameras + 1):
            # Constrói o caminho completo para cada vídeo
            video_path = os.path.join(video_base_path, f"camera_{i}_cutted.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
            self.video_captures.append(cap)
        
        print(f"{self.num_cameras} vídeos carregados com sucesso de '{video_base_path}'.")

    def _calib_img(self, image: np.ndarray, cam_index: int) -> np.ndarray:
        """Aplica a calibração para remover a distorção da imagem."""
        npzCalib = self.calib_data[cam_index]
        undistort_img = cv2.undistort(image, npzCalib['K'], npzCalib['dist'], None, npzCalib['nK'])
        if 'roi' in npzCalib:
            x, y, w, h = npzCalib['roi']
            return undistort_img[0:h, 0:w]
        return undistort_img

    def process_next_frame(self):
        """
        Lê o próximo frame de cada vídeo, executa a detecção e retorna os resultados.
        """
        frames = []
        all_frames_read = True
        for cap in self.video_captures:
            ret, frame = cap.read()
            if not ret:
                all_frames_read = False
                break
            frames.append(frame)

        # Se algum vídeo terminou, encerra o processamento
        if not all_frames_read:
            return None, None
            
        annotations = []
        # Processa cada frame (calibração e detecção)
        for i, frame in enumerate(frames):
            # 1. Corrige a distorção da imagem
            calibrated_frame = self._calib_img(frame, i)
            frames[i] = calibrated_frame # Atualiza a lista com o frame calibrado
            
            # 2. Detecta os esqueletos no frame
            results = self.detector.detect(calibrated_frame)
            
            # 3. Converte os resultados para o formato ObjectAnnotations
            #    (usando o método da sua classe SkeletonsDetector)
            obs_annotations = self.detector.to_object_annotations(results, calibrated_frame.shape)
            annotations.append(obs_annotations)
            
        return frames, annotations

    def release(self):
        """Libera os objetos de captura de vídeo."""
        for cap in self.video_captures:
            cap.release()
        print("Recursos de vídeo liberados.")