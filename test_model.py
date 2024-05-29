import torch

from dataloader.dataloader import get_scene
from gaussians.gaussian_model import GaussianModel
from render.render import render, render_image

config = {
    "background": torch.tensor([0.0, 0.0, 0.0], device="cuda"),
    "sh_degree": 3
}

scene_path = ""
scene = get_scene(scene_path)

from_iteration = 7000
checkpoint = torch.load(f"{scene_path}/gaussian_model/model/" + str(from_iteration) + ".model")

gaussian_model = GaussianModel(config["sh_degree"], scene.nerf_normalization["radius"])
gaussian_model.restore(checkpoint["gaussian"])

camera = scene.get_camera(20)
render_output = render(camera=camera,
                       gaussian=gaussian_model,
                       bg_color=config["background"])
image = torch.clamp(render_output.rgb, min=0, max=1.0) * -1
render_image(image)
