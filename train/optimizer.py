import torch

from gaussians.gaussian_model import GaussianModel
from .utils import get_expon_lr_func


def add_lr_update_method(obj):
    def update_lr(self, step):
        for p in self.param_groups:
            if "lr_scheduler" in p:
                p["lr"] *= p["lr_scheduler"](step)

    obj.update_lr = update_lr.__get__(obj)
    return obj


def get_optimizer(gaussian_model: GaussianModel,
                  initial_configuration: dict | None = None) -> torch.optim.Optimizer:
    if initial_configuration is None:
        initial_configuration = {"position_lr_init": 0.00016,
                                 "position_lr_delay_mult": 0.01,
                                 "position_lr_max_steps": 30_000,
                                 "position_lr_final": 0.0000016,
                                 "feature_lr": 0.0025,
                                 "opacity_lr": 0.05,
                                 "scaling_lr": 0.005,
                                 "rotation_lr": 0.001, }
    xyz_scheduler = get_expon_lr_func(
        lr_init=initial_configuration["position_lr_init"] * gaussian_model.spatial_lr_scale,
        lr_final=initial_configuration["position_lr_final"] * gaussian_model.spatial_lr_scale,
        lr_delay_mult=initial_configuration["position_lr_delay_mult"],
        max_steps=initial_configuration["position_lr_max_steps"])

    params = [
        {'params': [gaussian_model.xyz],
         'lr': initial_configuration["position_lr_init"] * gaussian_model.spatial_lr_scale,
         "name": "xyz",
         "lr_scheduler": lambda step: xyz_scheduler(step)
         },
        {'params': [gaussian_model.features_dc], 'lr': initial_configuration["feature_lr"], "name": "features_dc"},
        {'params': [gaussian_model.features_rest], 'lr': initial_configuration["feature_lr"] / 20.0,
         "name": "features_rest"},
        {'params': [gaussian_model.opacity], 'lr': initial_configuration["opacity_lr"], "name": "opacity"},
        {'params': [gaussian_model.scaling], 'lr': initial_configuration["scaling_lr"], "name": "scaling"},
        {'params': [gaussian_model.rotation], 'lr': initial_configuration["rotation_lr"], "name": "rotation"}
    ]

    return add_lr_update_method(torch.optim.Adam(params, lr=0.0, eps=1e-15))
