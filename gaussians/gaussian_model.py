import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from dataloader.point_cloud import BasicPointCloud
from .spherical_harmonics import RGB2SH
from .utils import inverse_sigmoid, distCUDA2


class GaussianModel(nn.Module):
    param_names = [
        "xyz",
        "features_dc",  # colors
        "features_rest",  # shs
        "scaling",
        "rotation",  # quaternions
        "opacity",
    ]

    def __init__(self, sh_degree: int, extent_radius: float):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.extent_radius = extent_radius

    def __len__(self):
        return len(self.xyz)

    @property
    def get_size(self):
        return torch.exp(self.scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotation)

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def increaseoneSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_parameters(self, state_dict: dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        parameters = {
            "xyz": nn.Parameter(fused_point_cloud.requires_grad_(True).cuda(), ),
            "features_dc": nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True).cuda()),
            "features_rest": nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True).cuda()),
            "scaling": nn.Parameter(scales.requires_grad_(True).cuda()),
            "rotation": nn.Parameter(rots.requires_grad_(True).cuda()),
            "opacity": nn.Parameter(opacities.requires_grad_(True).cuda()),
        }

        self.set_parameters(parameters)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1] * self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1] * self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def generate_point_cloud(self):
        xyz = self.xyz.detach().cpu()
        colors = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        return point_cloud

    def capture(self):
        return (self.active_sh_degree,
                self.xyz,
                self.features_dc,
                self.features_rest,
                self.scaling,
                self.rotation,
                self.opacity,
                self.spatial_lr_scale,)

    def restore(self, model_args):
        (self.active_sh_degree,
         self.xyz,
         self.features_dc,
         self.features_rest,
         self.scaling,
         self.rotation,
         self.opacity,
         self.spatial_lr_scale,) = model_args
