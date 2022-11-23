import json
import logging
import os
from copy import deepcopy
from typing import Optional

import mrcfile
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


class Manipulator:
    """Model Manipulator
    Args:
        classifiers: pretrained classifiers
        num_samples: number of samples
        device: device on which to evaluate model
        save_path: save path
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        num_samples: int,
        device: torch.device,
        save_path: Optional[str],
        n_images: int,
        extract_shape: int,
        seed: int,
        topk: int,
        latent_codes_path: Optional[str],
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        self.topk = topk

        # Device
        self.device = device

        # Generator
        self.generator = generator

        # n images & Number of samples & extract_shape flag
        self.seed = seed
        self.n_images = n_images
        self.num_samples = num_samples
        self.extract_shape = extract_shape

        # Save path and latent code
        self.save_path = to_absolute_path(save_path)
        self.latent_codes_path = (
            to_absolute_path(latent_codes_path) if latent_codes_path else None
        )

        # Latent codes
        self.latent_codes = np.load(os.path.join(self.save_path, "output.npz"))[
            "latent_codes"
        ]

        # Attributes
        self.attributes = pd.read_csv(os.path.join(self.save_path, "attributes.csv"))

        # Metrics and Feature Importance
        self.metrics = json.load(open(os.path.join(self.save_path, "metrics.json")))
        best_parameters = os.path.join(self.save_path, "best_parameters.json")
        self.best_parameters = (
            json.load(open(best_parameters))
            if os.path.isfile(best_parameters)
            else None
        )
        self.feature_importances = json.load(
            open(os.path.join(self.save_path, "results.json"))
        )

        suffix = (
            "results"
            if self.latent_codes_path is None
            else f"results_{self.latent_codes_path.split('/')[-2]}"
        )
        self.out_dir = os.path.join(self.save_path, suffix)

    def manipulate_latent(self, code, attribute):
        """Manipulates latent codes"""
        topk = (
            self.best_parameters[attribute]["best_k"]
            if self.best_parameters is not None
            else self.topk
        )
        logits = self.attributes[attribute].values
        feature_importance = self.feature_importances[attribute]
        feature_importance = self.generator.manipulated_indices(feature_importance)

        # positive direction
        codes = self.latent_codes[logits.argsort()[-self.n_images :][::-1]]
        dist = cosine_distances(code, codes)
        feats = codes[dist.argmin()]
        positive_code = deepcopy(code)

        for d, val in zip(feature_importance[:topk], feats[feature_importance]):
            positive_code[:, d] = val

        # negative direction
        codes = self.latent_codes[logits.argsort()[: self.n_images]]
        dist = cosine_distances(code, codes)
        feats = codes[dist.argmin()]
        negative_code = deepcopy(code)

        for d, val in zip(feature_importance[:topk], feats[feature_importance]):
            negative_code[:, d] = val

        return [negative_code, code, positive_code]

    def manipulate(self):
        """Manipulates images"""

        codes = self.generator.get_codes(
            self.generator.sample_latent(self.num_samples, seed=self.seed)
        )

        if self.latent_codes_path is not None:
            codes = torch.from_numpy(
                np.load(self.latent_codes_path)["latent_codes"]
            ).to(self.device)

        os.makedirs(self.out_dir, exist_ok=True)
        self.attributes = self.attributes[list(self.feature_importances.keys())]

        for attribute in tqdm(self.attributes.columns):
            if not self.metrics[attribute]["is_valid"]:
                continue
            imgs = []
            for i in range(codes.shape[0]):
                code = codes[i].view(1, -1).cpu().numpy()
                manipulated_codes = self.manipulate_latent(code, attribute)
                manipulated_imgs = []
                for manipulated_code, direction in zip(
                    manipulated_codes, ["negative", "original", "positive"]
                ):
                    manipulated_code = torch.from_numpy(manipulated_code).to(
                        self.device
                    )
                    img = self.generator.synthesize(
                        manipulated_code, h_angle=0, v_angle=0
                    )
                    manipulated_imgs.append(img[0])
                    if i == 0 and self.extract_shape:
                        voxel_grid = self.generator.extract_shape(manipulated_code)
                        shape_dir = os.path.join(self.out_dir, attribute)
                        os.makedirs(shape_dir, exist_ok=True)
                        with mrcfile.new_mmap(
                            os.path.join(shape_dir, f"{self.seed}_{direction}.mrc"),
                            overwrite=True,
                            shape=voxel_grid.shape,
                            mrc_mode=2,
                        ) as mrc:
                            mrc.data[:] = voxel_grid

                img = np.concatenate(manipulated_imgs, axis=1)
                imgs.append(img)

            imgs = (np.concatenate(imgs, axis=0) * 255).astype(np.uint8)
            img = Image.fromarray(imgs)
            img.save(os.path.join(self.out_dir, f"{attribute}.png"))
