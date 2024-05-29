import torch
from torch import nn
import numpy as np
from .utils import getWorld2View2, getProjectionMatrix


class CustomCamera:
    def __init__(self, width, height, fov_y, fov_x, znear, zfar, translation_matrix: np.array,
                 rotation_matrix: np.array):
        self.width = width
        self.height = height
        self.fov_x = fov_y
        self.fov_y = fov_x
        self.znear = znear
        self.zfar = zfar
        self.translation_matrix = translation_matrix
        self.rotation_matrix = rotation_matrix

        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        self._update()

    def _update(self):
        self.world_view_transform = torch.tensor(
            getWorld2View2(
                self.rotation_matrix, self.translation_matrix, self.trans, self.scale)).transpose(0, 1).to("cuda")
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fov_x,
                                                     fovY=self.fov_y).transpose(0, 1).to("cuda")
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to("cuda")

    def update_position(self, x_increment: float, y_increment: float, z_increment: float):
        self.translation_matrix += np.array([x_increment, y_increment, z_increment])
        self._update()

    def update_rotation(self, x_angle, y_angle, z_angle):  # degrees

        x_angle = np.radians(x_angle)
        y_angle = np.radians(y_angle)
        z_angle = np.radians(z_angle)

        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(x_angle), -np.sin(x_angle)],
                                      [0, np.sin(x_angle), np.cos(x_angle)]])

        rotation_matrix_y = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                                      [0, 1, 0],
                                      [-np.sin(y_angle), 0, np.cos(y_angle)]])

        rotation_matrix_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                      [np.sin(z_angle), np.cos(z_angle), 0],
                                      [0, 0, 1]])

        combined_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

        # Apply the rotation
        self.rotation_matrix = combined_rotation_matrix @ self.rotation_matrix.transpose(0, 1)
        self._update()

    def update_rotation_2(self, roll, pitch, yaw):
        # Isolate the rotation matrix (top-left 3x3 of extrinsics matrix)
        R = self.rotation_matrix

        # Calculate angles
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6  # Check for gimbal lock

        if not singular:
            pitch_ = np.arctan2(-R[2, 0], sy)
            yaw_ = np.arctan2(R[1, 0], R[0, 0])
            roll_ = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock case
            pitch_ = np.arctan2(-R[2, 0], sy)
            yaw_ = np.arctan2(-R[0, 2], R[1, 1])
            roll_ = 0

        # Convert angles to degrees
        roll_, pitch_, yaw_ = np.rad2deg([roll_, pitch_, yaw_])

        roll, pitch, yaw = np.deg2rad([roll + roll_, pitch + pitch_, yaw + yaw_])

        # Rotation matrices around the X, Y, and Z axis
        rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(roll), -np.sin(roll), 0],
                [0, np.sin(roll), np.cos(roll), 0],
                [0, 0, 0, 1],
            ]
        )

        ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), 0],
                [0, 1, 0, 0],
                [-np.sin(pitch), 0, np.cos(pitch), 0],
                [0, 0, 0, 1],
            ]
        )
        rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, 0],
                [np.sin(yaw), np.cos(yaw), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Combined rotation matrix (Z * Y * X)
        r = rz @ ry @ rx

        # Extrinsics matrix (Rotation * Translation)
        self.rotation_matrix = r[:3, :3]
        self._update()


class Camera(nn.Module):
    device = "cuda"
    zfar = 100.0
    znear = 0.01

    def __init__(self, id: str, image_path: str, image: torch.Tensor, fl_x: float, fl_y: float, fov_x: float,
                 fov_y: float, width: int, height: int, translation_matrix: np.array, rotation_matrix: np.array):
        super(Camera, self).__init__()
        self.id = id
        self.image_path = image_path
        self.image = image.to(self.device)
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.width = width
        self.height = height
        self.translation_matrix = translation_matrix
        self.rotation_matrix = rotation_matrix

        # self.original_image = image.clamp(0.0, 1.0).to(self.device)
        # self.original_image *= torch.ones((1, self.height, self.width), device=self.device)

        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        self.world_view_transform = torch.tensor(
            getWorld2View2(self.rotation_matrix, self.translation_matrix, self.trans, self.scale)).transpose(0, 1).to(
            self.device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fov_x,
                                                     fovY=self.fov_y).transpose(0, 1).to(self.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.device)
