import json
import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from src.criteria import id_loss, moco_loss

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class Tuner:
    """Model Tuner
    Args:
        generator: pretrained generator
        save_path: save path
        identity_threshold: threshold for identity loss
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        save_path: Optional[str],
        identity_threshold: Optional[float],
        topks: Optional[list],
        num_samples: Optional[int],
        n_images: Optional[int],
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        self.device = generator.device

        # Top K's and identity threshold
        self.topks = topks
        self.n_images = n_images
        self.num_samples = num_samples
        self.identity_threshold = identity_threshold

        if (
            "ffhq" in generator.class_name.lower()
            or "celeb" in generator.class_name.lower()
            or "faces" in generator.class_name.lower()
        ):
            self.identity_loss = (
                id_loss.IDLoss(
                    Path(__file__).parent / "encoders/checkpoints/model_ir_se50.pth"
                )
                .to(self.device)
                .eval()
            )
        else:
            self.identity_loss = (
                moco_loss.MocoLoss(
                    Path(__file__).parent
                    / "encoders/checkpoints/moco_v2_800ep_pretrain.pth"
                )
                .to(self.device)
                .eval()
            )

        # Generator
        self.generator = generator

        # Save path
        self.save_path = to_absolute_path(save_path)

        # Latent codes
        self.latent_codes = generator.prepare_latent_codes(
            np.load(os.path.join(self.save_path, "output.npz"))["latent_codes"]
        )

        # Attributes
        self.attributes = pd.read_csv(os.path.join(self.save_path, "attributes.csv"))

        # Metrics and Feature Importance
        self.metrics = json.load(open(os.path.join(self.save_path, "metrics.json")))
        self.feature_importances = json.load(
            open(os.path.join(self.save_path, "results.json"))
        )

    def calculate_id_loss(self, code, attribute):
        """calcualte identity loss for latent codes"""
        logits = self.attributes[attribute].values
        feature_importance = self.feature_importances[attribute]
        feature_importance = self.generator.manipulated_indices(feature_importance)

        tensor = self.generator.synthesize(code, return_tensor=True).to(self.device)

        # code = self.generator.prepare_latent_codes(code)

        neg_pos, neg_neu, pos_neu = [], [], []

        for topk in self.topks:
            # positive direction
            codes = self.latent_codes[logits.argsort()[-self.n_images :][::-1]]

            code_numpy = self.generator.prepare_latent_codes(
                code.view(1, -1).cpu().numpy()
            )

            dist = cosine_distances(code_numpy, codes)
            feats = torch.from_numpy(codes[dist.argmin()]).to(self.device)
            positive_code = deepcopy(code)

            for d, val in zip(feature_importance[:topk], feats[feature_importance]):
                positive_code[:, d] = val

            positive_tensor = self.generator.synthesize(
                positive_code, return_tensor=True
            ).to(self.device)

            # negative direction
            codes = self.latent_codes[logits.argsort()[: self.n_images]]
            dist = cosine_distances(code_numpy, codes)
            feats = torch.from_numpy(codes[dist.argmin()]).to(self.device)
            negative_code = deepcopy(code)

            for d, val in zip(feature_importance[:topk], feats[feature_importance]):
                negative_code[:, d] = val

            negative_tensor = self.generator.synthesize(
                negative_code, return_tensor=True
            ).to(self.device)

            neg_neu.append(
                self.identity_loss(positive_tensor, tensor, tensor)[0].item()
            )
            neg_pos.append(
                self.identity_loss(positive_tensor, negative_tensor, negative_tensor)[
                    0
                ].item()
            )
            pos_neu.append(
                self.identity_loss(negative_tensor, tensor, tensor)[0].item()
            )

        return neg_pos, neg_neu, pos_neu

    def tune_parameters(self):
        """Tunes the parameters"""

        self.attributes = self.attributes[list(self.feature_importances.keys())]

        results = {}
        for attr in tqdm(self.attributes.columns):
            results[attr] = {"topks": list(self.topks)}
            NP, NN, PN = [], [], []
            for _ in range(self.num_samples):
                code = self.generator.get_codes(self.generator.sample_latent(1))
                neg_pos, neg_neu, pos_neu = self.calculate_id_loss(code, attr)
                NP.append(neg_pos)
                NN.append(neg_neu)
                PN.append(pos_neu)

            neg_neu = np.stack(NN).mean(0)
            pos_neu = np.stack(PN).mean(0)
            neg_pos = np.stack(NP).mean(0)

            results[attr]["neg_neu"] = [float(f"{val:.4f}") for val in neg_neu.tolist()]
            results[attr]["pos_neu"] = [float(f"{val:.4f}") for val in pos_neu.tolist()]
            results[attr]["neg_pos"] = [float(f"{val:.4f}") for val in neg_pos.tolist()]

            arr = np.stack((neg_pos, pos_neu, neg_neu)).mean(0)
            results[attr]["mean"] = [float(f"{val:.4f}") for val in arr.tolist()]

            diff = arr - self.identity_threshold
            diff[diff > 0] = 1
            diff_arr = np.absolute(diff)
            argmin = diff_arr.argmin()

            best_topk = self.topks[int(argmin)]

            results[attr]["best_k"] = int(f"{best_topk}")

        json.dump(
            results,
            open(os.path.join(self.save_path, "best_parameters.json"), "w"),
            indent=4,
        )
