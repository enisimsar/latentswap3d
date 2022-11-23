import copy
import logging
import math
import os
import sys
from pathlib import Path

import wget
import numpy as np
import torch

import numpy as np
import torch
from torchvision.transforms import Resize

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


class VolumeGANGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "ffhq", is_inversion: bool = False
    ) -> None:
        super(VolumeGANGenerator, self).__init__()

        module_path = Path(__file__).parent / "volumegan"
        sys.path.insert(1, str(module_path.resolve()))

        self.device = device
        self.class_name = class_name
        self.is_inversion = is_inversion

        self.load_model()
        self.resize = Resize((256, 256))

    def load_model(self):
        module_path = Path(__file__).parent / "volumegan"
        sys.path.insert(1, str(module_path.resolve()))

        from configs import build_config, CONFIG_POOL
        from models import build_model

        checkpoints = {
            "ffhq": "https://www.dropbox.com/s/ygwhufzwi2vb2t8/volumegan_ffhq256.pth?dl=0"
        }

        network_pkl = checkpoints[self.class_name]

        print('Loading networks from "%s"...' % network_pkl)
        device = self.device

        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"volumegan/{self.class_name}.pth"

        os.makedirs(str(Path(checkpoint_root) / f"volumegan/"), exist_ok=True)

        if not os.path.isfile(str(checkpoint)):
            wget.download(
                network_pkl,
                str(checkpoint),
            )

        state = torch.load(checkpoint, map_location="cpu")
        G = build_model(**state["model_kwargs_init"]["generator_smooth"])
        G.load_state_dict(state["models"]["generator_smooth"], strict=True)

        self.model = G.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.ps_cfg = state["model_kwargs_init"]["generator"]["ps_cfg"]

    def get_codes(self, z: torch.Tensor, grad_enabled: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            codes = self.model.mapping(z, None)["wp"]

            return codes.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = 0,
        v_angle: float = 0,
        return_tensor=False,
    ) -> np.array:
        codes = codes.view(codes.shape[0], -1, 512)

        with torch.no_grad():
            w_avg = self.model.w_avg.reshape(1, -1, 512)[:, :8]
            codes[:, :8] = w_avg.lerp(codes[:, :8], 0.7)

            ps_kwargs = {}

            ps_kwargs["horizontal_stddev"] = 0
            ps_kwargs["vertical_stddev"] = 0
            ps_kwargs["num_steps"] = self.ps_cfg["num_steps"] * 3
            ps_kwargs["horizontal_mean"] = self.ps_cfg["horizontal_mean"] + h_angle
            ps_kwargs["vertical_mean"] = self.ps_cfg["vertical_mean"] + v_angle

            nerf_w = codes[:, : self.model.num_nerf_layers]
            cnn_w = codes[:, self.model.num_nerf_layers :]
            feature2d = self.model.nerf_synthesis(w=nerf_w, ps_kwargs=ps_kwargs)
            synthesis_results = self.model.synthesis(
                feature2d,
                cnn_w,
                lod=None,
                noise_mode="const",
                fused_modulate=False,
                impl="cuda",
                fp16_res=None,
            )

            img = synthesis_results["image"]

            img = self.resize(img)

            if return_tensor:
                return img

            img = img.detach().cpu().numpy()
            img = (img + 1) * 255 / 2
            img = np.clip(img + 0.5, 0, 255).astype(np.uint8) / 255
            img = img.transpose(0, 2, 3, 1)
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
        pass

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
        raise NotImplementedError
