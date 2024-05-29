import torch

from train.train import train

title = "Aprendizaje profundo para reconstrucción 3D de escenas: Neural Radiance Fields y Gaussian Splatting"
config = {
    "iterations": 10_000,  # 30000

    "start_densification": 500,  # 500
    "densification_interval": 100,
    "finish_densification": 7_500,  # 15000

    "reset_opacity_interval": 2_500,  # 3000
    "reset_opacity_value": 0.01,
    "max_num_gaussians": 3_000_000,

    "sh_degree": 3,
    "background": torch.tensor([0.0, 0.0, 0.0], device="cuda"),

    "big_point_px_radius": 20,  # 50
    "big_point_cov_extent": 0.1,
    "min_opacity_pruning": 0.005,

    "densify_grad_threshold": 0.0002,
    "densify_split_clone_threshold": 0.01,
    "densify_split_scale_factor": torch.tensor(0.8, device="cuda"),
}

scene_path = ""
load_from_iteration = None

train(scene_path=scene_path, config=config, from_iteration=load_from_iteration)
