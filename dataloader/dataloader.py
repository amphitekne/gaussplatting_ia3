import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .camera import Camera
from .point_cloud import get_point_cloud
from .scene import SceneData
from .utils import qvec2rotmat, focal2fov


# def load_image(image_path: str) -> torch.Tensor:
#     return read_image(image_path) * -1
def load_image(image_path: str) -> torch.Tensor:
    # return read_image(image_path) * -1
    _image = Image.open(image_path)
    _img = np.array(_image)
    return torch.from_numpy(_img[None, ...]).type(torch.uint8).squeeze().permute(2, 0, 1)


def read_colmap(scene_path: str | Path) -> tuple:
    def get_points() -> dict:
        with open(os.path.join(scene_path, "colmap_text", "points3D.txt"), "r") as file:
            text_lines = [line.split("\n")[0].split(" ") for line in file]
        text_lines = text_lines[3:]
        point_cloud = {
            line[0]: {"xyz": np.array(line[1:4]).astype(np.float32), "color": np.array(line[4:7]).astype(np.integer),
                      "error": line[7]} for line in text_lines}
        return point_cloud

    def get_images_data() -> dict:
        with open(os.path.join(scene_path, "colmap_text", "images.txt"), "r") as file:
            text_lines = [line.split("\n")[0].split(" ") for line in file]
        text_lines = text_lines[4::2]
        images_data = {
            line[0]: {"qvec": np.array(line[1:5]).astype(np.float32), "tvec": np.array(line[5:8]).astype(np.float32),
                      "camera_id": line[8], "image_name": line[9]} for line in text_lines}
        return images_data

    def get_camera_data() -> dict:
        with open(os.path.join(scene_path, "colmap_text", "cameras.txt"), "r") as file:
            text_lines = [line.split("\n")[0].split(" ") for line in file]
        text_lines = text_lines[3:]
        camera_data = {
            line[0]: {"model": line[1],
                      "width": int(line[2]),
                      "height": int(line[3]),
                      "fl_x": float(line[4]),
                      "fl_y": float(line[5]),
                      "c_x": float(line[6]),
                      "c_y": float(line[7]),
                      }
            for line in text_lines}
        return camera_data

    camera_data = get_camera_data()
    images_data = get_images_data()
    points = get_points()

    cameras = []
    for image_id, data in tqdm(images_data.items()):
        camera_id = data["camera_id"]
        image_path = os.path.join(scene_path, "images", data["image_name"])
        image = load_image(image_path)
        camera = Camera(id=camera_id,
                        image_path=image_path,
                        image=image,
                        width=image.shape[2],
                        height=image.shape[1],
                        fl_x=camera_data[camera_id]["fl_x"],
                        fl_y=camera_data[camera_id]["fl_y"],
                        fov_x=focal2fov(camera_data[camera_id]["fl_x"], camera_data[camera_id]["width"]),
                        fov_y=focal2fov(camera_data[camera_id]["fl_y"], camera_data[camera_id]["height"]),
                        rotation_matrix=np.transpose(qvec2rotmat(data["qvec"])),
                        translation_matrix=np.transpose(data["tvec"]),
                        )
        cameras.append(camera)

    # Initial Point Cloud
    xyz = np.stack([content["xyz"] for key, content in points.items()]).astype(np.float32)
    colors = np.stack([content["color"] for key, content in points.items()]).astype(np.uint8)

    return cameras, get_point_cloud(xyz, colors)


def get_scene(scene_path: str | Path) -> SceneData:
    cameras, point_cloud = read_colmap(scene_path=scene_path)
    return SceneData(cameras, point_cloud)
