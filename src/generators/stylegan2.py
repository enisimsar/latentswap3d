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
import torch.nn as nn
from torchvision.transforms import Resize

from src.generators.abstract import Generator
from src.utils.model_utils import download_ckpt

logger = logging.getLogger(__name__)


def block_forward(
    self,
    x,
    img,
    ws,
    shapes,
    force_fp32=False,
    fused_modconv=None,
    apply_affine=True,
    **layer_kwargs,
):
    from torch_utils import misc
    from torch_utils import persistence
    from torch_utils.ops import conv2d_resample
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import fma

    misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
    w_iter = iter(ws.unbind(dim=1))
    dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
    memory_format = (
        torch.channels_last
        if self.channels_last and not force_fp32
        else torch.contiguous_format
    )
    if fused_modconv is None:
        # this value will be treated as a constant
        with misc.suppress_tracer_warnings():
            fused_modconv = (not self.training) and (
                dtype == torch.float32 or int(x.shape[0]) == 1
            )

    # Input.
    if self.in_channels == 0:
        x = self.const.to(dtype=dtype, memory_format=memory_format)
        x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
    else:
        misc.assert_shape(
            x, [None, self.in_channels, self.resolution // 2, self.resolution // 2]
        )
        x = x.to(dtype=dtype, memory_format=memory_format)

    # Main layers.
    if self.in_channels == 0:
        if apply_affine:
            x = self.conv1(
                x,
                next(w_iter)[..., : shapes[0]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
        else:
            affine = self.conv1.affine
            self.conv1.affine = nn.Identity()
            x = self.conv1(
                x,
                next(w_iter)[..., : shapes[0]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            self.conv1.affine = affine
    elif self.architecture == "resnet":
        y = self.skip(x, gain=np.sqrt(0.5))
        if apply_affine:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        else:
            affine = self.conv0.affine
            self.conv0.affine = nn.Identity()
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            self.conv0.affine = affine

        if apply_affine:
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
        else:
            affine = self.conv1.affine
            self.conv1.affine = nn.Identity()
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            self.conv1.affine = affine
        x = y.add_(x)
    else:
        if apply_affine:
            x = x = self.conv0(
                x,
                next(w_iter)[..., : shapes[0]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
        else:
            affine = self.conv0.affine
            self.conv0.affine = nn.Identity()
            x = x = self.conv0(
                x,
                next(w_iter)[..., : shapes[0]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            self.conv0.affine = affine

        if apply_affine:
            x = self.conv1(
                x,
                next(w_iter)[..., : shapes[1]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
        else:
            affine = self.conv1.affine
            self.conv1.affine = nn.Identity()
            x = self.conv1(
                x,
                next(w_iter)[..., : shapes[1]],
                fused_modconv=fused_modconv,
                **layer_kwargs,
            )
            self.conv1.affine = affine

    # ToRGB.
    if img is not None:
        misc.assert_shape(
            img, [None, self.img_channels, self.resolution // 2, self.resolution // 2]
        )
        img = upfirdn2d.upsample2d(img, self.resample_filter)
    if self.is_last or self.architecture == "skip":
        if apply_affine:
            y = self.torgb(
                x, next(w_iter)[..., : shapes[2]], fused_modconv=fused_modconv
            )
        else:
            affine = self.torgb.affine
            self.torgb.affine = nn.Identity()
            y = self.torgb(
                x, next(w_iter)[..., : shapes[2]], fused_modconv=fused_modconv
            )
            self.torgb.affine = affine
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

    assert x.dtype == dtype
    assert img is None or img.dtype == torch.float32
    return x, img


class StyleGAN2Generator(Generator):
    def __init__(
        self, device: str, class_name: str = "ffhq", is_inversion: bool = False
    ) -> None:
        super(StyleGAN2Generator, self).__init__()

        module_path = Path(__file__).parent / "stylegan2"
        sys.path.insert(1, str(module_path.resolve()))

        self.truncation = 0.7
        self.resolution = 1024

        self.device = device
        self.class_name = class_name
        self.is_inversion = is_inversion

        # ffhq
        self.temp_shapes = [
            (512, 512, 512),
            (512, 512, 512),
            (512, 512, 512),
            (512, 512, 512),
            (512, 512, 512),
            (512, 256, 256),
            (256, 128, 128),
            (128, 64, 64),
            (64, 32, 32),
        ]

        self.load_model()
        self.resize = Resize((256, 256))

    def load_model(self):
        import dnnlib
        import legacy

        checkpoints = {
            "ffhq": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            "afhqcat": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl",
            "metfaces": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
            "afhqdog": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
        }

        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"stylegan2/{self.class_name}"

        with dnnlib.util.open_url(
            checkpoints[self.class_name], cache_dir=checkpoint
        ) as f:
            self.model = legacy.load_network_pkl(f)["G_ema"].to(self.device)
            for i in self.model.parameters():
                i.requires_grad = False

        self.model.eval()

    def _get_codes(self, z: torch.Tensor) -> torch.Tensor:
        from torch_utils import misc

        label = torch.zeros([1, self.model.c_dim], device=self.device).requires_grad_()

        ws = self.model.mapping(z, label, truncation_psi=self.truncation)

        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(
                ws, [None, self.model.synthesis.num_ws, self.model.synthesis.w_dim]
            )
            ws = ws.to(torch.float32)

            w_idx = 0
            for res in self.model.synthesis.block_resolutions:
                block = getattr(self.model.synthesis, f"b{res}")
                # ws = 18 (9 resolution/block x 2 w) x 512
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        styles = torch.zeros(
            1, 26, 512, device=self.device
        )  # for s space we have 26 (skip first layer) instead of 3 layers * 2 w
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(self.model.synthesis.block_resolutions, block_ws):
            block = getattr(self.model.synthesis, f"b{res}")

            if res == 4:  # first layer we have 2 outputs
                temp_shape = (
                    block.conv1.affine.weight.shape[0],
                    block.conv1.affine.weight.shape[0],
                    block.torgb.affine.weight.shape[0],
                )
                styles[0, :1, :] = block.conv1.affine(cur_ws[0, :1, :])
                styles[0, 1:2, :] = block.torgb.affine(cur_ws[0, 1:2, :])
                styles_idx += 2
            else:  # 3 layers 2 convs and torgb
                temp_shape = (
                    block.conv0.affine.weight.shape[0],
                    block.conv1.affine.weight.shape[0],
                    block.torgb.affine.weight.shape[0],
                )
                styles[
                    0, styles_idx : styles_idx + 1, : temp_shape[0]
                ] = block.conv0.affine(cur_ws[0, :1, :])
                styles[
                    0, styles_idx + 1 : styles_idx + 2, : temp_shape[1]
                ] = block.conv1.affine(cur_ws[0, 1:2, :])
                styles[
                    0, styles_idx + 2 : styles_idx + 3, : temp_shape[2]
                ] = block.torgb.affine(cur_ws[0, 2:3, :])
                styles_idx += 3

            temp_shapes.append(temp_shape)
        styles = styles.detach()

        return styles.view(z.shape[0], -1)

    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        styles = []

        for zs in z:
            styles.append(self._get_codes(zs.view(1, -1)))

        return torch.cat(styles, dim=0)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = None,
        v_angle: float = None,
        return_tensor=False,
    ) -> np.array:
        codes = codes.view(codes.shape[0], 26, 512)

        with torch.no_grad():
            styles_idx = 0
            x = img = None
            noise_mode = "const"
            for k, res in enumerate(self.model.synthesis.block_resolutions):
                block = getattr(self.model.synthesis, f"b{res}")
                if res == 4:
                    x, img = block_forward(
                        block,
                        x,
                        img,
                        codes[:, styles_idx : styles_idx + 2, :],
                        self.temp_shapes[k],
                        noise_mode=noise_mode,
                        apply_affine=False,
                    )
                    styles_idx += 2
                else:
                    x, img = block_forward(
                        block,
                        x,
                        img,
                        codes[:, styles_idx : styles_idx + 3, :],
                        self.temp_shapes[k],
                        noise_mode=noise_mode,
                        apply_affine=False,
                    )
                    styles_idx += 3

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
            ) / 255.0
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
