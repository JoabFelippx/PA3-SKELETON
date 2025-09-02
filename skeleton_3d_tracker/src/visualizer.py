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
        # MUDANÇA: Dicionário para guardar cores consistentes para cada ID
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

    # MUDANÇA: O método update agora aceita 'tracked_trajectories'
    def update(self, image_to_draw: np.ndarray, tracked_skeletons_with_ids: list, extrinsic_matrices: list, tracked_trajectories: list):
        self.ax.clear()
        self.ax.set_xlim(-4.5, 4.5); self.ax.set_ylim(-4.5, 4.5); self.ax.set_zlim(0, 3)
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        
        self._draw_camera_axes(extrinsic_matrices)
        
        for person in tracked_skeletons_with_ids:
            track_id = person['id']
            skeleton = person['skeleton']
            if not skeleton: continue

            # MUDANÇA: Define uma cor consistente para o ID
            if track_id not in self.id_colors:
                self.id_colors[track_id] = self.color_map(len(self.id_colors) % 10)
            color = self.id_colors[track_id]

            points = np.array(list(skeleton.values()))
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, color=color, label=f"Track ID: {track_id}")
            if 0 in skeleton:
                head_pos = skeleton[0]
                self.ax.text(head_pos[0], head_pos[1], head_pos[2] + 0.1, f"ID {track_id}", color=color, fontsize=12, fontweight='bold')

        # MUDANÇA: Novo loop para desenhar as linhas de trajetória
        for traj_data in tracked_trajectories:
            track_id = traj_data['id']
            trajectory = np.array(traj_data['trajectory'])
            
            if track_id in self.id_colors and trajectory.shape[0] > 1:
                color = self.id_colors[track_id]
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=5)

        if tracked_skeletons_with_ids:
            self.ax.legend()
            
        plot_img_rgba = self._fig_to_numpy()
        plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

        h1, w1, _ = image_to_draw.shape
        h2, w2, _ = plot_img_bgr.shape
        if h1 != h2:
            new_w = int(w2 * h1 / h2)
            plot_img_bgr = cv2.resize(plot_img_bgr, (new_w, h1))

        combined_view = np.hstack((image_to_draw, plot_img_bgr))
        cv2.imshow('Multi-camera 3D Skeleton Tracking', combined_view)