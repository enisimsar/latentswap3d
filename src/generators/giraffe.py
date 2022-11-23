import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.generators.abstract import Generator

module_path = Path(__file__).parent / "giraffe"
sys.path.insert(1, str(module_path.resolve()))

from im2scene import config
from im2scene.checkpoints import CheckpointIO

logger = logging.getLogger(__name__)


class GiraffeGenerator(Generator):
    def __init__(
        self, device: str, class_name: str = "ffhq", object_type: str = "both"
    ) -> None:
        super(GiraffeGenerator, self).__init__()

        self.device = device
        self.class_name = class_name

        self.load_model()

        if object_type not in ["foreground", "background", "both"]:
            raise Exception("Type is not supported")

        self.object_type = object_type
        self.shape_keys = []
        if object_type in ["foreground", "both"]:
            self.shape_keys.append("z_shape_obj")
            self.shape_keys.append("z_app_obj")

        if object_type in ["background", "both"]:
            self.shape_keys.append("z_shape_bg")
            self.shape_keys.append("z_app_bg")

        self.shapes = [
            ("z_shape_obj", self.model.get_n_boxes() * 256),
            ("z_app_obj", self.model.get_n_boxes() * 256),
            ("z_shape_bg", 128),
            ("z_app_bg", 128),
        ]

    def load_config(self, path):
        """Loads config file.
        Args:
            path (str): path to config file
            default_path (bool): whether to use default path
        """
        # Load configuration from file itself
        with open(path, "r") as f:
            cfg_special = yaml.load(f, Loader=yaml.Loader)

        # Check if we should inherit from a config
        inherit_from = cfg_special.get("inherit_from")

        # If yes, load this config first as default
        # If no, use the default_path
        if inherit_from is not None:
            cfg = config.load_config(
                path.replace("_pretrained", ""),
                default_path=str(
                    Path(__file__).parent / f"giraffe/configs/default.yaml"
                ),
            )
        else:
            cfg = dict()

        # Include main configuration
        config.update_recursive(cfg, cfg_special)

        return cfg

    def load_model(self):
        checkpoint_root = Path(__file__).parent / "checkpoints"
        checkpoint = Path(checkpoint_root) / f"giraffe"

        config_path = (
            Path(__file__).parent
            / f"giraffe/configs/256res/{self.class_name}_256_pretrained.yaml"
        )

        cfg = self.load_config(str(config_path))

        self.model = config.get_model(cfg, device=self.device)
        checkpoint_io = CheckpointIO(checkpoint, model=self.model)
        checkpoint_io.load(cfg["test"]["model_file"])

        self.model = self.model.generator_test

    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size = list(z.values())[0].shape[0]
            codes = torch.cat(list(z.values()), dim=1).view(batch_size, -1)

            return codes

    def synthesize(
        self,
        codes: torch.Tensor,
        h_angle: float = 0,
        v_angle: float = 0,
        return_tensor=False,
    ) -> np.array:
        if self.class_name == "cars" and h_angle == 0:
            h_angle = 0.25
        with torch.no_grad():
            batch_size = codes.shape[0]
            keys = [k for k, _ in self.shapes]
            vals = [v for _, v in self.shapes]

            codes = torch.split(codes, vals, dim=1)
            codes = {k: v * 0.65 for k, v in zip(keys, codes)}

            codes["z_shape_obj"] = codes["z_shape_obj"].view(
                batch_size, self.model.get_n_boxes(), 256
            )
            codes["z_app_obj"] = codes["z_app_obj"].view(
                batch_size, self.model.get_n_boxes(), 256
            )

            latent_codes = (
                codes["z_shape_obj"],
                codes["z_app_obj"],
                codes["z_shape_bg"],
                codes["z_app_bg"],
            )

            bg_rotation = self.model.get_random_bg_rotation(batch_size)

            # Set Camera
            camera_matrices = self.model.get_camera(
                batch_size=batch_size, val_u=0.5 + h_angle, val_v=0.5 + v_angle
            )
            s_val = [[0, 0, 0] for i in range(self.model.get_n_boxes())]
            t_val = [[0.5, 0.5, 0.5] for i in range(self.model.get_n_boxes())]
            r_val = [0.0 for i in range(self.model.get_n_boxes())]
            s, t, _ = self.model.get_transformations(s_val, t_val, r_val, batch_size)

            r = [0.5 + h_angle]
            r = self.model.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]

            img = self.model(
                batch_size,
                latent_codes,
                camera_matrices,
                transformations,
                bg_rotation,
                mode="val",
            )

            if return_tensor:
                return img

            AA = img.reshape(img.size(0), -1)
            AA -= AA.min(1, keepdim=True)[0]
            AA /= AA.max(1, keepdim=True)[0]
            img = AA.view(img.size(0), img.size(1), img.size(2), img.size(3))

            img = img.permute(0, 2, 3, 1).cpu().numpy()

        return img

    def synthesize_inversion(self, codes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        """Samples random codes from the latent space"""
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        torch.manual_seed(seed)
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = self.model.get_latent_codes(
            batch_size, tmp=1.0
        )

        latent_codes = {}

        latent_codes["z_shape_obj"] = z_shape_obj.view(batch_size, -1)
        latent_codes["z_app_obj"] = z_app_obj.view(batch_size, -1)
        latent_codes["z_shape_bg"] = z_shape_bg.view(batch_size, -1)
        latent_codes["z_app_bg"] = z_app_bg.view(batch_size, -1)

        return latent_codes

    def mean_latent(self) -> torch.Tensor:
        pass

    def _get_indices(self, shape_keys):
        indices = []
        current_indices = 0
        for (shape_key, dim) in self.shapes:
            if shape_key in shape_keys:
                indices.extend(range(current_indices, current_indices + dim))
            current_indices += dim
        return indices

    def prepare_latent_codes(self, codes):
        indices = self._get_indices(self.shape_keys)
        return codes[:, indices]

    def manipulated_indices(self, indices):
        mapped_indices = self._get_indices([key for key, _ in self.shapes])
        return np.array([mapped_indices[i] for i in indices])
