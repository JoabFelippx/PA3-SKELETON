# visualizer.py

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=-110)
        self.id_colors = {}
        self.color_map = plt.get_cmap('tab10')

    def _fig_to_numpy(self) -> np.ndarray:
        self.fig.canvas.draw()
        rgba_buffer = self.fig.canvas.buffer_rgba()
        return np.asarray(rgba_buffer)

    def _draw_camera_axes(self, extrinsic_matrices: list, axis_length: float = 0.5):
        axis_colors = ['r', 'g', 'b']
        for i, rt in enumerate(extrinsic_matrices):
            R = rt[:, :3]
            t = rt[:, 3]
            cam_center = -R.T @ t
            cam_axes_in_world = R.T
            self.ax.scatter(*cam_center, s=80, c='black', marker='o')
            self.ax.text(*cam_center, f' Cam {i+1}', color='black', fontsize=12)
            for j in range(3):
                axis_start = cam_center
                axis_end = cam_center + cam_axes_in_world[:, j] * axis_length
                self.ax.plot([axis_start[0], axis_end[0]],
                             [axis_start[1], axis_end[1]],
                             [axis_start[2], axis_end[2]],
                             color=axis_colors[j], linewidth=3)

    def update(self, images: list, skeletons_to_visualize: list, extrinsic_matrices: list):
        #print(skeletons_to_visualize)
        self.ax.clear()
        self.ax.set_xlim(-4.5, 4.5); self.ax.set_ylim(-4.5, 4.5); self.ax.set_zlim(0, 3)
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')

        self._draw_camera_axes(extrinsic_matrices)

        self.id_colors.clear()

        for person in skeletons_to_visualize:

            person_id = person['id']
            skeleton = person['skeleton_3d']
            if not skeleton: continue

            if person_id not in self.id_colors:
                self.id_colors[person_id] = self.color_map(len(self.id_colors) % 10)
            color = self.id_colors[person_id]

            points = np.array(list(skeleton.values()))
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, color=color, label=f"ID (Frame): {person_id}")
            if 0 in skeleton:
                head_pos = skeleton[0]
                self.ax.text(head_pos[0], head_pos[1], head_pos[2] + 0.1, f"ID {person_id}", color=color, fontsize=12, fontweight='bold')


        if skeletons_to_visualize:
            self.ax.legend()

        plot_img_rgba = self._fig_to_numpy()
        plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)


        largura = 960
        altura = 540
        dim = (largura, altura)

        img1_redimensionada = cv2.resize(images[0], dim, interpolation=cv2.INTER_AREA)
        img2_redimensionada = cv2.resize(images[1], dim, interpolation=cv2.INTER_AREA)
        img3_redimensionada = cv2.resize(images[2], dim, interpolation=cv2.INTER_AREA)
        img4_redimensionada = cv2.resize(images[3], dim, interpolation=cv2.INTER_AREA)

        linha_superior = cv2.hconcat([img1_redimensionada, img2_redimensionada])
        linha_inferior = cv2.hconcat([img3_redimensionada, img4_redimensionada])
        imagem_grade = cv2.vconcat([linha_superior, linha_inferior])

        h2, w2, _ = plot_img_bgr.shape
        imagem_grade = cv2.resize(imagem_grade, (h2, w2))


        combined_view = np.hstack((imagem_grade, plot_img_bgr))
        cv2.imshow('Multi-camera 3D Skeleton Reconstruction', combined_view)







