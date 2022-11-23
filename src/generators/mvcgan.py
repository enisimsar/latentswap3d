import copy
import logging
import math
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch
from torchvision.transforms import Resize

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


def staged_forward_with_frequencies(
    self,
    truncated_frequencies,
    truncated_phase_shifts,
    truncated_mapping_codes,
    img_size=256,
    fov=12,
    ray_start=0.88,
    ray_end=1.12,
    num_steps=16,
    h_stddev=0,
    v_stddev=0,
    h_mean=0,
    v_mean=0,
    sample_dist=None,
    lock_view_dependence=False,
    nerf_noise=0,
    last_back=False,
    white_back=False,
    clamp_mode="relu",
    output_size=256,
    alpha=1.0,
    is_inversion=False,
    **kwargs,
):
    """
    Similar to forward but used for inference.
    Calls the model sequencially using max_batch_size to limit memory usage.
    """
    from generators.volumetric_rendering import (
        rgb_feat_integration,
        transform_sampled_points,
    )

    batch_size = truncated_frequencies.shape[0]

    self.generate_avg_frequencies()

    with torch.set_grad_enabled(is_inversion):
        # get_initial_rays_trig
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, img_size, device=self.device),
            torch.linspace(1, -1, img_size, device=self.device),
        )
        x = x.T.flatten()
        y = y.T.flatten()
        z_coord = -torch.ones_like(x, device=self.device) / np.tan(
            (2 * math.pi * fov / 360) / 2
        )

        rays_d_cam = torch.stack([x, y, z_coord], -1)
        rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
        rays_d_cam = rays_d_cam / rays_d_norm  # (height*width, 3)

        z_vals = (
            torch.linspace(ray_start, ray_end, num_steps, device=self.device)
            .reshape(1, num_steps, 1)
            .repeat(img_size * img_size, 1, 1)
        )
        points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

        points_cam = torch.stack(batch_size * [points_cam])
        z_vals = torch.stack(batch_size * [z_vals])
        rays_d_cam = torch.stack(batch_size * [rays_d_cam]).to(self.device)

        (
            primary_points,
            primary_z_vals,
            primary_ray_directions,
            primary_ray_origins,
            primary_pitch,
            primary_yaw,
            primary_cam2world_matrix,
        ) = transform_sampled_points(
            points_cam,
            z_vals,
            rays_d_cam,
            h_stddev=h_stddev,
            v_stddev=v_stddev,
            h_mean=h_mean,
            v_mean=v_mean,
            device=self.device,
            mode=sample_dist,
        )

        primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
        primary_ray_directions_expanded = primary_ray_directions_expanded.expand(
            -1, -1, num_steps, -1
        )
        primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(
            batch_size, img_size * img_size * num_steps, 3
        )
        primary_points = primary_points.reshape(
            batch_size, img_size * img_size * num_steps, 3
        )

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(
                transformed_ray_directions_expanded
            )
            transformed_ray_directions_expanded[..., -1] = -1

        rgb_feat_dim = 256
        (
            primary_output,
            primary_rgb_feat,
        ) = self.siren.forward_with_frequencies_phase_shifts(
            primary_points,
            truncated_frequencies,
            truncated_phase_shifts,
            ray_directions=primary_ray_directions_expanded,
        )

        primary_output = primary_output.reshape(
            batch_size, img_size * img_size, num_steps, 4
        )
        primary_rgb_feat = primary_rgb_feat.reshape(
            batch_size, img_size * img_size, num_steps, rgb_feat_dim
        )

        (
            primary_initial_rgb,
            primary_rgb_feat_maps,
            primary_depth,
            _,
        ) = rgb_feat_integration(
            primary_output,
            primary_rgb_feat,
            primary_z_vals,
            device=self.device,
            white_back=white_back,
            last_back=last_back,
            clamp_mode=clamp_mode,
            noise_std=nerf_noise,
        )

        primary_depth_map = (
            primary_depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
        )

        primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(
            batch_size, img_size, img_size, rgb_feat_dim
        )
        primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

        rgb_feat_maps = primary_rgb_feat_maps

        pixels = self.decoder(
            truncated_mapping_codes, rgb_feat_maps, img_size, output_size, alpha
        )

    return pixels, primary_depth_map


class MVCGANGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "CelebAHQ", is_inversion: bool = False
    ) -> None:
        super(MVCGANGenerator, self).__init__()

        module_path = Path(__file__).parent / "mvcgan"
        sys.path.insert(1, str(module_path.resolve()))

        import curriculums

        self.device = device
        self.class_name = class_name
        self.is_inversion = is_inversion

        curriculum = getattr(
            curriculums,
            class_name if class_name in ["CelebAHQ", "Cat"] else class_name.upper(),
        )
        curriculum["num_steps"] = curriculum[0]["num_steps"]
        curriculum["img_size"] = 64
        curriculum["output_size"] = 512
        curriculum["psi"] = 0.5
        curriculum["v_stddev"] = 0
        curriculum["h_stddev"] = 0
        curriculum["lock_view_dependence"] = False
        curriculum["last_back"] = curriculum.get("eval_last_back", False)
        curriculum["nerf_noise"] = 0

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

        self.load_model()

        self.curriculum = {
            key: value for key, value in curriculum.items() if type(key) is str
        }

        self.resize = Resize((256, 256))

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self, outfile):
        checkpoints = {
            "CelebAHQ": [
                "https://drive.google.com/uc?export=download&id=1k2jOYxtzxme6BUgxVDhpKxIzRm_LK5zW",
                "https://drive.google.com/uc?export=download&id=1OuVM7G8tF11xRAyDXIrJWCE3bhvZgKvN",
            ],
            "Cat": [
                "https://drive.google.com/uc?export=download&id=1yMzmztgpnQ4kPuWjYBlGNqwF-R2xxsWc",
                "https://drive.google.com/uc?export=download&id=1i4QkHgahPtJx82Ho2tR0H98iMvwCHLV1",
            ],
            "FFHQ": [
                "https://drive.google.com/uc?export=download&id=16WS5PwEsVNZUmTzrIb2-J8DZ00c_LDny",
                "https://drive.google.com/uc?export=download&id=1-b8Wasd1w4P7tdyGfu1bTnrm627u5ejo",
            ],
        }

        urls = checkpoints[self.class_name]
        for name, url in zip(["generator.pth", "ema.pth"], urls):
            out_path = os.path.join(outfile, name)
            if not os.path.exists(out_path):
                gdown.download(url, out_path, quiet=False)

    def load_model(self):
        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"mvcgan/{self.class_name}"

        if not checkpoint.is_dir():
            os.makedirs(checkpoint, exist_ok=True)
            self.download_checkpoint(checkpoint)

        model_path = os.path.join(checkpoint, "generator.pth")

        generator = torch.load(model_path, map_location=torch.device(self.device))
        ema_file = model_path.replace("generator.pth", "ema.pth")
        ema = torch.load(ema_file)
        ema.copy_to(generator.parameters())
        generator.set_device(self.device)
        generator.eval()

        self.model = generator

    def get_codes(self, z: torch.Tensor, grad_enabled: bool = False) -> torch.Tensor:
        self.model.generate_avg_frequencies()

        with torch.set_grad_enabled(grad_enabled):

            (
                raw_frequencies,
                raw_phase_shifts,
                raw_avg_mapping_codes,
            ) = self.model.siren.mapping_network(z)

            codes = torch.cat(
                (raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes), axis=1
            )
            return codes.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = None,
        v_angle: float = None,
        return_tensor=False,
    ) -> np.array:
        splits = torch.split(codes, [2304, 2304, 256], dim=1)
        raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = (
            splits[0],
            splits[1],
            splits[2],
        )

        if h_angle is not None and v_angle is not None:
            self.curriculum["num_steps"] = 48
            self.curriculum["sample_dist"] = None
            self.curriculum["h_mean"] = math.pi / 2 + h_angle
            self.curriculum["v_mean"] = math.pi / 2 + v_angle

        with torch.no_grad():
            truncated_frequencies = self.model.avg_frequencies + self.curriculum[
                "psi"
            ] * (raw_frequencies - self.model.avg_frequencies)
            truncated_phase_shifts = self.model.avg_phase_shifts + self.curriculum[
                "psi"
            ] * (raw_phase_shifts - self.model.avg_phase_shifts)
            truncated_mapping_codes = self.model.avg_mapping_codes + self.curriculum[
                "psi"
            ] * (raw_avg_mapping_codes - self.model.avg_mapping_codes)

            img, _ = staged_forward_with_frequencies(
                self.model,
                truncated_frequencies,
                truncated_phase_shifts,
                truncated_mapping_codes,
                **self.curriculum,
            )

            if return_tensor:
                return img

            img = self.resize(img)
            img = ((img + 1) / 2).float()
            img = img.clamp_(0, 1)

            from torchvision.utils import make_grid

            grid = make_grid(
                img, nrow=1, padding=0, pad_value=1, normalize=True
            ).permute(1, 2, 0)

        return torch.unsqueeze(grid, 0).detach().cpu().numpy()

    def synthesize_inversion(self, codes: torch.Tensor, img_size=128) -> torch.Tensor:
        splits = torch.split(codes, [2304, 2304, 256], dim=1)
        raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = (
            splits[0],
            splits[1],
            splits[2],
        )

        options = copy.deepcopy(self.curriculum)
        options["h_mean"] = torch.tensor(math.pi / 2).to(self.device) + self.h_mean
        options["v_mean"] = torch.tensor(math.pi / 2).to(self.device) + self.v_mean
        options["sample_dist"] = None
        options["last_back"] = False
        options["hierarchical_sample"] = False
        options["lock_view_dependence"] = False

        truncated_frequencies = self.model.avg_frequencies + self.curriculum["psi"] * (
            raw_frequencies - self.model.avg_frequencies
        )
        truncated_phase_shifts = self.model.avg_phase_shifts + self.curriculum[
            "psi"
        ] * (raw_phase_shifts - self.model.avg_phase_shifts)
        truncated_mapping_codes = self.model.avg_mapping_codes + self.curriculum[
            "psi"
        ] * (raw_avg_mapping_codes - self.model.avg_mapping_codes)

        tensor_img, _ = staged_forward_with_frequencies(
            self.model,
            truncated_frequencies,
            truncated_phase_shifts,
            truncated_mapping_codes,
            is_inversion=self.is_inversion,
            **options,
        )

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
        return torch.randn((batch_size, 256), device=self.device)

    def mean_latent(self) -> torch.Tensor:
        """Returns the mean of the latent space"""
        (
            mean_frequencies,
            mean_phase_shifts,
            mean_mapping_codes,
        ) = self.model.generate_avg_frequencies()
        codes = torch.cat(
            (mean_frequencies, mean_phase_shifts, mean_mapping_codes), axis=1
        )
        return codes.view(1, -1)

    def extract_shape(
        self,
        codes: torch.Tensor,
        max_batch=200000,
        voxel_resolution=256,
        voxel_origin=[0, 0, 0],
        cube_length=0.25,
    ):
        head = 0
        samples, voxel_origin, _ = create_samples(
            voxel_resolution, voxel_origin, cube_length
        )
        samples = samples.to(self.device)
        sigmas = torch.zeros(
            (samples.shape[0], samples.shape[1], 1), device=self.device
        )
        rgb = torch.zeros((samples.shape[0], samples.shape[1], 3), device=self.device)

        transformed_ray_directions_expanded = torch.zeros(
            (samples.shape[0], max_batch, 3), device=self.device
        )
        transformed_ray_directions_expanded[..., -1] = -1

        splits = torch.split(codes, [2304, 2304], dim=1)
        raw_frequencies, raw_phase_shifts = splits[0], splits[1]

        with torch.no_grad():
            truncated_frequencies = self.model.avg_frequencies + self.curriculum[
                "psi"
            ] * (raw_frequencies - self.model.avg_frequencies)
            truncated_phase_shifts = self.model.avg_phase_shifts + self.curriculum[
                "psi"
            ] * (raw_phase_shifts - self.model.avg_phase_shifts)

        with torch.no_grad():
            while head < samples.shape[1]:
                coarse_output = self.model.siren.forward_with_frequencies_phase_shifts(
                    samples[:, head : head + max_batch],
                    truncated_frequencies,
                    truncated_phase_shifts,
                    ray_directions=transformed_ray_directions_expanded[
                        :, : samples.shape[1] - head
                    ],
                ).reshape(samples.shape[0], -1, 4)
                sigmas[:, head : head + max_batch] = coarse_output[:, :, -1:]
                rgb[:, head : head + max_batch] = coarse_output[:, :, :-1]
                head += max_batch

        sigmas = (
            sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution))
            .cpu()
            .numpy()
        )

        return sigmas

    def extract_depth_map(self, codes: torch.Tensor, h_angle=0, v_angle=0) -> np.array:
        splits = torch.split(codes, [2304, 2304, 256], dim=1)
        raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = (
            splits[0],
            splits[1],
            splits[2],
        )

        if h_angle is not None and v_angle is not None:
            self.curriculum["h_mean"] = math.pi / 2 + h_angle
            self.curriculum["v_mean"] = math.pi / 2 + v_angle

        with torch.no_grad():
            truncated_frequencies = self.model.avg_frequencies + self.curriculum[
                "psi"
            ] * (raw_frequencies - self.model.avg_frequencies)
            truncated_phase_shifts = self.model.avg_phase_shifts + self.curriculum[
                "psi"
            ] * (raw_phase_shifts - self.model.avg_phase_shifts)
            truncated_mapping_codes = self.model.avg_mapping_codes + self.curriculum[
                "psi"
            ] * (raw_avg_mapping_codes - self.model.avg_mapping_codes)

            _, depth_map = staged_forward_with_frequencies(
                self.model,
                truncated_frequencies,
                truncated_phase_shifts,
                truncated_mapping_codes,
                is_inversion=self.is_inversion,
                **self.curriculum,
            )

        return Resize((256, 256))(depth_map)
