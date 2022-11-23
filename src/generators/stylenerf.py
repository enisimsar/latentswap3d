import copy
import logging
import math
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch

from copy import deepcopy as dc

from torchvision.transforms import Resize

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


class StyleNeRFGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "ffhq", is_inversion: bool = False
    ) -> None:
        super(StyleNeRFGenerator, self).__init__()

        module_path = Path(__file__).parent / "stylenerf"
        sys.path.insert(1, str(module_path.resolve()))

        self.device = device
        self.class_name = class_name
        self.is_inversion = is_inversion

        relative_range_u_scale = 1.0
        self.synthesis_kwargs = {
            "truncation_psi": 1.0,
            "noise_mode": "const",
            "render_option": None,
            "n_steps": 8,
            "render_program": None,  # "rotation_camera",
            "return_cameras": True,
        }

        self.load_model()
        self.resize = Resize((256, 256))

    def load_model(self):
        import dnnlib
        import legacy
        from renderer import Renderer

        checkpoints = {
            "ffhq": "https://huggingface.co/facebook/stylenerf-ffhq-config-basic/resolve/main/ffhq_512.pkl",
        }

        network_pkl = checkpoints[self.class_name]

        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"stylenerf/{self.class_name}"

        device = self.device

        with dnnlib.util.open_url(network_pkl, cache_dir=checkpoint) as f:
            network = legacy.load_network_pkl(f)
            G = network["G_ema"].to(device)  # type: ignore
            D = network["D"].to(device)

        print('Loading networks from "%s"...' % network_pkl)

        # Labels.
        class_idx = None
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail(
                    "Must specify class label with --class when using a conditional network"
                )
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print(
                    "warn: --class=lbl ignored when running on an unconditional network"
                )

        self.label = label

        # avoid persistent classes...
        from training.networks import Generator
        from torch_utils import misc

        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
        G2 = Renderer(G2, D, program=self.synthesis_kwargs["render_program"])

        self.model = G2
        self.model.generator.eval()

        for param in self.model.generator.parameters():
            param.requires_grad = False

    def get_codes(self, z: torch.Tensor, grad_enabled: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            codes = self.model.generator.mapping(z, self.label, **self.synthesis_kwargs)

            return codes.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = 0,
        v_angle: float = 0,
        return_tensor=False,
    ) -> np.array:
        batch_size = codes.shape[0]
        codes = codes.view(batch_size, -1, 512)

        camera_matrices = self.model.generator.synthesis.get_camera(
            batch_size=batch_size,
            mode=[0.5 + h_angle, 0.5 + v_angle, 0.5],
            device=self.device,
        )

        params = dc(self.synthesis_kwargs)
        params["camera_matrices"] = camera_matrices
        params["not_render_background"] = True

        with torch.no_grad():
            # Apply truncation.
            with torch.autograd.profiler.record_function("truncate"):
                codes = self.model.generator.mapping.w_avg.lerp(codes, 0.7)
            img = self.model(styles=codes, **params)

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
        return torch.randn((batch_size, self.model.generator.z_dim), device=self.device)

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
