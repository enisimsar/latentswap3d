import copy
import logging
import math
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch

from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


class PiGANGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "CelebA", is_inversion: bool = False
    ) -> None:
        super(PiGANGenerator, self).__init__()

        self.device = device
        self.class_name = class_name

        module_path = Path(__file__).parent / "pi-GAN"
        sys.path.insert(1, str(module_path.resolve()))
        import curriculums

        curriculum = getattr(
            curriculums, class_name if class_name == "CelebA" else class_name.upper()
        )
        curriculum["num_steps"] = curriculum[0]["num_steps"] * 2
        curriculum["img_size"] = 256
        curriculum["psi"] = 0.7
        curriculum["v_stddev"] = 0
        curriculum["h_stddev"] = 0
        curriculum["last_back"] = True
        curriculum["lock_view_dependence"] = True
        curriculum["last_back"] = curriculum.get("eval_last_back", False)
        curriculum["nerf_noise"] = 0
        curriculum["hierarchical_sample"] = True
        curriculum["max_batch_size"] = 2000000

        self.load_model()

        if is_inversion:
            self.h_mean = torch.zeros(1).to(device)
            self.v_mean = torch.zeros(1).to(device)

            for param in self.model.parameters():
                param.requires_grad = True

        self.curriculum = {
            key: value for key, value in curriculum.items() if type(key) is str
        }

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self, outfile):
        checkpoints = {
            "CelebA": "https://drive.google.com/uc?id=1bRB4-KxQplJryJvqyEa8Ixkf_BVm4Nn6",
            "Cats": "https://drive.google.com/uc?id=1WBA-WI8DA7FqXn7__0TdBO0eO08C_EhG",
        }

        url = checkpoints[self.class_name]
        gdown.download(url, f"{str(outfile)}.zip", quiet=False)

        import zipfile

        with zipfile.ZipFile(f"{str(outfile)}.zip", "r") as zip_ref:
            checkpoint_root = Path(__file__).parent / "checkpoints"
            checkpoint = Path(checkpoint_root) / f"pigan"
            zip_ref.extractall(str(checkpoint))

    def load_model(self):
        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"pigan/{self.class_name}"

        if not checkpoint.is_dir():
            os.makedirs(checkpoint.parent, exist_ok=True)
            self.download_checkpoint(checkpoint)

        model_path = os.path.join(checkpoint, "generator.pth")

        generator = torch.load(model_path, map_location=torch.device(self.device))
        ema_file = model_path.replace("generator.pth", "ema.pth")
        ema = torch.load(ema_file)
        ema.copy_to(generator.parameters())
        generator.set_device(self.device)
        generator.eval()

        self.model = generator

        for param in self.model.parameters():
            param.requires_grad = False

    def get_codes(self, z: torch.Tensor, grad_enabled: bool = False) -> torch.Tensor:
        self.model.generate_avg_frequencies()

        with torch.set_grad_enabled(grad_enabled):
            raw_frequencies, raw_phase_shifts = self.model.siren.mapping_network(z)

            codes = torch.cat((raw_frequencies, raw_phase_shifts), axis=1)
            return codes.view(z.shape[0], -1)

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = None,
        v_angle: float = None,
        return_tensor=False,
    ) -> np.array:
        splits = torch.split(codes, [2304, 2304], dim=1)
        raw_frequencies, raw_phase_shifts = splits[0], splits[1]

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

            img, _ = self.model.staged_forward_with_frequencies(
                truncated_frequencies, truncated_phase_shifts, **self.curriculum
            )

            if return_tensor:
                return img

            AA = img.view(img.size(0), -1)
            AA -= AA.min(1, keepdim=True)[0]
            AA /= AA.max(1, keepdim=True)[0]
            img = AA.view(img.size(0), img.size(1), img.size(2), img.size(3))

            img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
        return img

    def synthesize_inversion(self, codes: torch.Tensor, img_size=128) -> torch.Tensor:
        splits = torch.split(codes, [2304, 2304], dim=1)
        raw_frequencies, raw_phase_shifts = splits[0], splits[1]
        options = copy.deepcopy(self.curriculum)
        options["num_steps"] = 24
        options["img_size"] = img_size
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

        tensor_img, _ = self.model.forward_with_frequencies(
            truncated_frequencies,
            truncated_phase_shifts,
            **options,
        )

        return tensor_img

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
        from extract_shapes import create_samples

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
        splits = torch.split(codes, [2304, 2304], dim=1)
        raw_frequencies, raw_phase_shifts = splits[0], splits[1]

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

            _, depth_map = self.model.staged_forward_with_frequencies(
                truncated_frequencies, truncated_phase_shifts, **self.curriculum
            )

        return depth_map
