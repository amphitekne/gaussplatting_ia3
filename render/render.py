# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
from collections import namedtuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from dataloader.dataloader import Camera
from gaussians.gaussian_model import GaussianModel

RasterOutputs = namedtuple(
    "RasterOutputs", ["rgb", "radii", "visibility_mask", "screenspace_points"]
)


def render(camera: Camera, gaussian: GaussianModel, bg_color: torch.Tensor,
           scaling_modifier: float = 1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussian.get_xyz, dtype=gaussian.get_xyz.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussian.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=gaussian.get_xyz,
        means2D=screenspace_points,
        shs=gaussian.get_features,
        colors_precomp=None,
        opacities=gaussian.get_opacity,
        scales=gaussian.get_size,
        rotations=gaussian.get_rotation,
        cov3D_precomp=None)

    rendered_image = torch.clamp(rendered_image, min=0, max=1.0)
    return RasterOutputs(rgb=rendered_image,
                         radii=radii,
                         visibility_mask=radii > 0,
                         screenspace_points=screenspace_points)


def render_image(render_tensor: torch.Tensor):
    transform = T.ToPILImage()
    img = transform(render_tensor)
    img.show()


def save_image(render_tensor: torch.Tensor, iteration: int, path: str):
    plt.imshow(render_tensor.permute(1, 2, 0).cpu())
    plt.axis('off')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{path}/{iteration}.png", dpi=600, bbox_inches='tight', pad_inches=0)
