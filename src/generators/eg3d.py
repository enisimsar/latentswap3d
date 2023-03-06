import copy
import logging
import math
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch
import torch.nn as nn

import numpy as np
import torch
from torchvision.transforms import Resize

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)

temp_shapes = [(512, None, 512), (512, 512, 512), (512, 512, 512), (512, 512, 512), (512, 512, 512), (512, 256, 256), (256, 128, 128)]

def block_forward(
    self,
    x,
    img,
    ws,
    shapes,
    force_fp32=False,
    fused_modconv=None,
    update_emas=False,
    apply_affine=True,
    **layer_kwargs,
):
    from torch_utils import misc
    from torch_utils import persistence
    from torch_utils.ops import conv2d_resample
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import fma
    _ = update_emas # unused
    misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
    w_iter = iter(ws.unbind(dim=1))

    if ws.device.type != 'cuda':
        force_fp32 = True
    dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
    memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
    if fused_modconv is None:
        fused_modconv = self.fused_modconv_default
    if fused_modconv == 'inference_only':
        fused_modconv = (not self.training)

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
                x, next(w_iter)[..., : shapes[2]], 
                fused_modconv=fused_modconv
            )
        else:
            affine = self.torgb.affine
            self.torgb.affine = nn.Identity()
            y = self.torgb(
                x, next(w_iter)[..., : shapes[2]], 
                fused_modconv=fused_modconv
            )
            self.torgb.affine = affine
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

    assert x.dtype == dtype
    assert img is None or img.dtype == torch.float32
    return x, img

def s_forward(self, ws, **block_kwargs):
    styles_idx = 0
    x = img = None
    noise_mode = "const"
    for k, res in enumerate(self.block_resolutions):
        block = getattr(self, f"b{res}")
        if res == 4:
            x, img = block_forward(
                block,
                x,
                img,
                ws[:, styles_idx : styles_idx + 2, :],
                temp_shapes[k],
                noise_mode=noise_mode,
                apply_affine=False,
            )
            styles_idx += 2
        else:
            x, img = block_forward(
                block,
                x,
                img,
                ws[:, styles_idx : styles_idx + 3, :],
                temp_shapes[k],
                noise_mode=noise_mode,
                apply_affine=False,
            )
            styles_idx += 3
    return img

def synthesis_forward(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
    cam2world_matrix = c[:, :16].view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)

    if neural_rendering_resolution is None:
        neural_rendering_resolution = self.neural_rendering_resolution
    else:
        self.neural_rendering_resolution = neural_rendering_resolution

    # Create a batch of rays for volume rendering
    ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

    # Create triplanes by running StyleGAN backbone
    N, M, _ = ray_origins.shape
    if use_cached_backbone and self._last_planes is not None:
        planes = self._last_planes
    else:
        planes = s_forward(self.backbone.synthesis, ws, update_emas=update_emas, **synthesis_kwargs)
    if cache_backbone:
        self._last_planes = planes

    # Reshape output into three 32-channel planes
    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    # Perform volume rendering
    feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

    # Reshape into 'raw' neural-rendered image
    H = W = self.neural_rendering_resolution
    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
    depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

    # Run superresolution to get final image
    rgb_image = feature_image[:, :3]
    sr_image = self.superresolution(rgb_image, feature_image, ws[:, -1:, :], noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

    return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}

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
        self.resize = Resize((512, 512))

        if is_inversion:
            self.h_mean = (
                # torch.distributions.uniform.Uniform(-0.7, 0.7)
                # .sample([1])
                # .to(device)
                torch.zeros(1).to(device)
            )
            self.v_mean = (
                # torch.distributions.uniform.Uniform(-0.7, 0.7)
                # .sample([1])
                # .to(device)
                torch.zeros(1).to(device)
            )

            # self.h_mean.requires_grad_()
            # self.v_mean.requires_grad_()

    def load_model(self):
        import dnnlib
        import legacy
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        checkpoints = {
            "ffhq": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl",
            "afhq": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/afhqcats512-128.pkl",
            "shapenet": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/shapenetcars128-64.pkl"
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

        self.model = G
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        from torch_utils import misc

        conditioning_params = torch.cat(
            [self.cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1
        )

        conditioning_params = conditioning_params.repeat(
            z.shape[0],
            1,
        )
        ws = self.model.mapping(
            z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14
        )


        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(
                ws, [None, self.model.backbone.synthesis.num_ws, self.model.backbone.synthesis.w_dim]
            )
            ws = ws.to(torch.float32)

            w_idx = 0
            for res in self.model.backbone.synthesis.block_resolutions:
                block = getattr(self.model.backbone.synthesis, f"b{res}")
                # ws = 18 (9 resolution/block x 2 w) x 512
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        styles = torch.zeros(
            z.shape[0], 20, 512, device=self.device
        )  # for s space we have 26 (skip first layer) instead of 3 layers * 2 w
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(self.model.backbone.synthesis.block_resolutions, block_ws):
            block = getattr(self.model.backbone.synthesis, f"b{res}")

            if res == 4:  # first layer we have 2 outputs
                temp_shape = (
                    None,
                    block.conv1.affine.weight.shape[0],
                    block.torgb.affine.weight.shape[0],
                )
                styles[:, 0, :] = block.conv1.affine(cur_ws[:, 0, :])
                styles[:, 1, :] = block.torgb.affine(cur_ws[:, 1, :])
                styles_idx += 2
            else:  # 3 layers 2 convs and torgb
                temp_shape = (
                    block.conv0.affine.weight.shape[0],
                    block.conv1.affine.weight.shape[0],
                    block.torgb.affine.weight.shape[0],
                )
                styles[
                    :, styles_idx, : temp_shape[0]
                ] = block.conv0.affine(cur_ws[:, 0, :])
                styles[
                    :, styles_idx + 1, : temp_shape[1]
                ] = block.conv1.affine(cur_ws[:, 1, :])
                styles[
                    :, styles_idx + 2, : temp_shape[2]
                ] = block.torgb.affine(cur_ws[:, 2, :])
                styles_idx += 3

            temp_shapes.append(temp_shape)
        styles = styles.detach()

        styles = torch.cat([styles, ws[:, -1:, :]], dim=1)

        shapes = []
        for l in temp_shapes:
            shapes.extend(l)
        shapes.append(512)

        shapes = [shape for shape in shapes if shape]

        cropped_styles = []
        for ind in range(len(shapes)):
            cropped_styles.append(styles[:, ind, :shapes[ind]])

        cropped_styles = torch.cat(cropped_styles, dim=1)
        return cropped_styles.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = 0,
        v_angle: float = 0,
        fov: float = 0,
        return_tensor=False,
    ) -> np.array:
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        all_codes = torch.zeros((codes.shape[0], 21, 512))

        shapes = []
        for l in temp_shapes:
            shapes.extend(l)
        shapes.append(512)
        shapes = [shape for shape in shapes if shape]

        splits = torch.split(codes, shapes, dim=1)
        for ind, split in enumerate(splits):
            all_codes[:, ind, :shapes[ind]] = split
        codes = all_codes.to(self.device)

        cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 + h_angle,
            3.14 / 2 + v_angle,
            torch.tensor([0, 0.065, 0.2], device=self.device),
            radius=2.7,
            device=self.device,
        )
        fov = fov + 18.837
        intrinsics = FOV_to_intrinsics(fov, device=self.device)

        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )

        with torch.set_grad_enabled(self.is_inversion):
            img = synthesis_forward(self.model, codes, camera_params)["image"]

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
        from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

        all_codes = torch.zeros((codes.shape[0], 21, 512))

        shapes = []
        for l in temp_shapes:
            shapes.extend(l)
        shapes.append(512)
        shapes = [shape for shape in shapes if shape]

        splits = torch.split(codes, shapes, dim=1)
        for ind, split in enumerate(splits):
            all_codes[:, ind, :shapes[ind]] = split
        codes = all_codes.to(self.device)

        cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 + self.h_mean,
            3.14 / 2 + self.v_mean,
            torch.tensor([0, 0.065, 0.2], device=self.device),
            radius=2.7,
            device=self.device,
        )
        fov = 18.837
        intrinsics = FOV_to_intrinsics(fov, device=self.device)

        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )

        with torch.set_grad_enabled(self.is_inversion):
            tensor_img = synthesis_forward(self.model, codes, camera_params)["image"]

            tensor_img = self.resize(tensor_img)

        return Resize((img_size, img_size))(tensor_img)

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
        samples = self.sample_latent(10000)
        codes = torch.mean(self.get_codes(samples), axis=0)
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
