import copy
import logging
import math
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch

import numpy as np
import torch
from torchvision.transforms import Resize

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


class EG3DGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "ffhq", is_inversion: bool = False
    ) -> None:
        super(EG3DGenerator, self).__init__()

        module_path = Path(__file__).parent / "eg3d/eg3d"
        sys.path.insert(1, str(module_path.resolve()))

        self.device = device
        self.class_name = class_name
        self.is_inversion = is_inversion

        self.load_model()
        self.resize = Resize((256, 256))

    def load_model(self):
        import dnnlib
        import legacy
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        checkpoints = {
            "ffhq": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl",
            "afhq": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/afhqcats512-128.pkl",
        }

        network_pkl = checkpoints[self.class_name]

        print('Loading networks from "%s"...' % network_pkl)
        device = self.device

        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"eg3d/{self.class_name}"

        with dnnlib.util.open_url(network_pkl, cache_dir=checkpoint) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

        self.cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2,
            3.14 / 2,
            torch.tensor([0, 0, 0.2], device=device),
            radius=2.7,
            device=device,
        )
        self.intrinsics = FOV_to_intrinsics(18.837, device=device)
        # self.intrinsics = FOV_to_intrinsics(16, device=device)

        self.model = G
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def get_codes(self, z: torch.Tensor, grad_enabled: bool = False) -> torch.Tensor:
        conditioning_params = torch.cat(
            [self.cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1
        )

        conditioning_params = conditioning_params.repeat(
            z.shape[0],
            1,
        )

        with torch.set_grad_enabled(grad_enabled):
            codes = self.model.mapping(
                z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14
            )

            return codes.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = 0,
        v_angle: float = 0,
        return_tensor=False,
    ) -> np.array:
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        codes = codes.view(codes.shape[0], -1, 512)
        cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 + h_angle,
            3.14 / 2 + v_angle,
            torch.tensor([0, 0.065, 0.2], device=self.device),
            radius=2.7,
            device=self.device,
        )
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)

        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )

        with torch.no_grad():
            img = self.model.synthesis(codes, camera_params)["image"]

            img = self.resize(img)

            if return_tensor:
                return img

            img = (
                (img.permute(0, 2, 3, 1) * 127.5 + 128)
                .clamp(0, 255)
                .to(torch.uint8)
                .detach()
                .cpu()
                .numpy()
            ) / 255
        return img

    def synthesize_inversion(self, codes: torch.Tensor, img_size=128) -> torch.Tensor:
        raise NotImplementedError

    def prelearnable_parameters(self) -> torch.nn.Parameter:
        return [self.h_mean, self.v_mean]  # None

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        """Samples random codes from the latent space"""
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        torch.manual_seed(seed)
        return torch.randn((batch_size, self.model.z_dim), device=self.device)

    def mean_latent(self) -> torch.Tensor:
        """Returns the mean of the latent space"""
        mean_frequencies, mean_phase_shifts = self.model.generate_avg_frequencies()
        codes = torch.cat((mean_frequencies, mean_phase_shifts), axis=1)
        return codes.view(1, -1)

    def extract_shape(
        self,
        codes: torch.Tensor,
        max_batch=200000,
        voxel_resolution=256,
        voxel_origin=[0, 0, 0],
        cube_length=0.25,
    ):
        raise NotImplementedError

    def extract_depth_map(self, codes: torch.Tensor, h_angle=0, v_angle=0) -> np.array:
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        codes = codes.view(codes.shape[0], -1, 512)
        cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 + h_angle,
            3.14 / 2 + v_angle,
            torch.tensor([0, 0.065, 0.2], device=self.device),
            radius=2.7,
            device=self.device,
        )
        intrinsics = FOV_to_intrinsics(18.837, device=self.device)

        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )

        with torch.no_grad():
            img = self.model.synthesis(codes, camera_params)["image_depth"]

            img = self.resize(img)

            img = -img
            img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
        return img.cpu()[0]
