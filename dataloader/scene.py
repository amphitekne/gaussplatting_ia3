import random
from dataclasses import dataclass

from .camera import Camera
from .point_cloud import BasicPointCloud
from .utils import getNerfppNorm


@dataclass
class SceneData:
    cameras: list[Camera]
    point_cloud: BasicPointCloud
    is_train_data: bool = True

    def __post_init__(self):
        if self.is_train_data:
            self.nerf_normalization = getNerfppNorm(self.cameras)

    def __len__(self):
        return len(self.cameras)

    def get_camera(self, idx: int) -> Camera:
        return self.cameras[idx]

    def split(self, test_size: float = 0.2):
        num_cameras = len(self.cameras)
        index = list(range(num_cameras))
        test_index = sorted(random.sample(index, int(num_cameras * test_size)))
        train_index = [i for i in index if i not in test_index]
        train_cameras = [self.cameras[camera_index] for camera_index in train_index]
        test_cameras = [self.cameras[camera_index] for camera_index in test_index]

        return (SceneData(cameras=train_cameras, point_cloud=self.point_cloud, is_train_data=True),
                SceneData(cameras=test_cameras, point_cloud=self.point_cloud, is_train_data=False))
