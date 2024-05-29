import os
from datetime import datetime
from random import randint
from time import time

import torch
from tqdm import tqdm

from dataloader.dataloader import get_scene, SceneData
from gaussians.gaussian_model import GaussianModel
from train.losses import compute_loss
from render.render import render, save_image
from train import metrics
from .densifier import Densifier
from .optimizer import get_optimizer


def datasets_txt(checkpoint_path: str, scene_train: SceneData, scene_test: SceneData):
    with open(os.path.join(checkpoint_path, "train_images.txt"), 'a') as file:
        for camera in scene_train.cameras:
            image_name = camera.image_path.split("\\")[-1]
            file.write(f"{image_name}" + '\n')
    with open(os.path.join(checkpoint_path, "test_images.txt"), 'a') as file:
        for camera in scene_test.cameras:
            image_name = camera.image_path.split("\\")[-1]
            file.write(f"{image_name}" + '\n')


def start_report(checkpoint_path: str, first_line: str):
    with open(os.path.join(checkpoint_path, "report.csv"), 'w') as file:
        pass
    with open(os.path.join(checkpoint_path, "report.csv"), 'a') as file:
        file.write(
            "iteration;timestamp;training_loss;testing_loss;training_ssim;testing_ssim;training_psnr;testing_psnr;training_lpips;testing_lpips;training_mse;testing_mse" + '\n')


def update_report(checkpoint_path: str, iteration: int,
                  training_loss: float, testing_loss: float,
                  training_ssim: float, testing_ssim: float,
                  training_psnr: float, testing_psnr: float,
                  training_lpips: float, testing_lpips: float,
                  training_mse: float, testing_mse: float,
                  ):
    with open(os.path.join(checkpoint_path, "report.csv"), 'a') as file:
        new_line = f"{iteration};{time()};{training_loss};{testing_loss};{training_ssim};{testing_ssim};{training_psnr};{testing_psnr};{training_lpips};{testing_lpips};{training_mse};{testing_mse}"
        file.write(new_line + '\n')


def load_model(scene: SceneData, scene_path: str, config: dict, from_iteration: int) -> tuple[
    GaussianModel, Densifier, torch.optim.Optimizer, int]:
    if from_iteration is None:
        gaussian_model = GaussianModel(config["sh_degree"], scene.nerf_normalization["radius"])
        gaussian_model.create_from_pcd(scene.point_cloud, spatial_lr_scale=0.1)
        optimizer = get_optimizer(gaussian_model)
        densifier = Densifier(gaussian_model, optimizer, config)
        initial_iteration = 1
    else:
        checkpoint = torch.load(f"{scene_path}/gaussian_model/model/" + str(from_iteration) + ".model")
        gaussian_model = GaussianModel(config["sh_degree"], scene.nerf_normalization["radius"])
        gaussian_model.restore(checkpoint["gaussian"])
        optimizer = get_optimizer(gaussian_model)
        densifier = Densifier(gaussian_model, optimizer, config)
        densifier.restore(checkpoint["densifier"])
        initial_iteration = from_iteration

    return gaussian_model, densifier, optimizer, initial_iteration


def check_directories(scene_path: str) -> tuple[str, str, str]:
    checkpoint_path = os.path.join(scene_path, "gaussian_model")
    images_directory = os.path.join(checkpoint_path, "images")
    model_directory = os.path.join(checkpoint_path, "model")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(images_directory):
        os.mkdir(images_directory)
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)
    return checkpoint_path, images_directory, model_directory


def test(scene: SceneData, gaussian_model: GaussianModel, config: dict) -> tuple[float, float, float]:
    cameras = scene.cameras
    losses = []
    psnr_losses = []
    mse_losses = []
    for camera in cameras:
        with torch.no_grad():
            render_output = render(camera=camera, gaussian=gaussian_model, bg_color=config["background"])
            ground_truth_image = camera.image.to(render_output.rgb.dtype) / 255.0
            losses.append(compute_loss(render_output.rgb, ground_truth_image).item())
            psnr_losses.append(metrics.PSNR(render_output.rgb, ground_truth_image))
            mse_losses.append(metrics.MSE(render_output.rgb, ground_truth_image))
    loss = sum(losses) / len(losses)
    psnr_loss = sum(psnr_losses) / len(psnr_losses)
    mse_loss = sum(mse_losses) / len(mse_losses)
    return loss, psnr_loss, mse_loss.item()


def train(scene_path, config: dict, from_iteration: None | int = None):
    scene = get_scene(scene_path)

    scene_train, scene_test = scene.split()
    checkpoint_path, images_directory, model_directory = check_directories(scene_path=scene_path)
    datasets_txt(checkpoint_path, scene_train, scene_test)
    gaussian_model, densifier, optimizer, initial_iteration = load_model(
        scene=scene,
        scene_path=scene_path,
        config=config,
        from_iteration=from_iteration
    )
    number_of_scene_images = len(scene_train)
    ema_10_loss = []
    ema_10_psnr = []
    ema_10_mse = []
    # Reference Image
    camera_ref = scene_test.get_camera(randint(0, len(scene_test) - 1))
    save_image(torch.clamp(camera_ref.image.to(torch.float32) / 255.0, min=0, max=1.0), "ref", images_directory)

    start_report(checkpoint_path=checkpoint_path, first_line=datetime.now().strftime("%H:%M:%S-%d/%m/%y"))
    for iteration in tqdm(range(initial_iteration, config["iterations"] + 1)):
        optimizer.zero_grad()

        camera = scene_train.get_camera(randint(0, number_of_scene_images - 1))

        if iteration % 1000 == 0:
            gaussian_model.increaseoneSHdegree()

        render_output = render(camera=camera,
                               gaussian=gaussian_model,
                               bg_color=config["background"])
        ground_truth_image = camera.image / 255.0
        loss = compute_loss(render_output.rgb, ground_truth_image)
        loss.backward()
        with torch.no_grad():
            ema_10_loss.append(loss.item())
            ema_10_psnr.append(metrics.PSNR(render_output.rgb, ground_truth_image).item())
            ema_10_mse.append(metrics.MSE(render_output.rgb, ground_truth_image).item())
            if len(ema_10_loss) > 10:
                ema_10_loss.pop(0)
                ema_10_psnr.pop(0)
                ema_10_mse.pop(0)
            if iteration % 10 == 0:
                print(
                    f"Iter {iteration} loss: {sum(ema_10_loss) / len(ema_10_loss)}. Num of points: {len(gaussian_model)}")

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer.update_lr(iteration)

            _ = densifier(iteration=iteration, render_output=render_output)

            if iteration % 100 == 0:
                torch.save({
                    "gaussian": gaussian_model.capture(),
                    "densifier": densifier.capture(),
                },
                    os.path.join(model_directory, str(iteration) + ".model"))
                render_output = render(
                    camera=camera_ref,
                    gaussian=gaussian_model,
                    bg_color=config["background"]
                )
                save_image(torch.clamp(render_output.rgb, min=0, max=1.0), iteration, images_directory)
                testing_loss, testing_psnr, testing_mse = test(scene=scene_test,
                                                               gaussian_model=gaussian_model,
                                                               config=config
                                                               )
                update_report(
                    checkpoint_path=checkpoint_path, iteration=iteration,
                    training_loss=sum(ema_10_loss) / len(ema_10_loss), testing_loss=testing_loss,
                    training_ssim=0, testing_ssim=0,
                    training_psnr=sum(ema_10_psnr) / len(ema_10_psnr), testing_psnr=testing_psnr,
                    training_lpips=0, testing_lpips=0,
                    training_mse=sum(ema_10_mse) / len(ema_10_mse), testing_mse=testing_mse,
                )
                torch.cuda.empty_cache()
