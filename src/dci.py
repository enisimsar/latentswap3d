import json
import logging
import math
import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path

from src.utils.dci_utils import DCI, DCI_Test


class DCIMetric:
    """Model Finder
    Args:
        generator: pretrained generator
        save_path: save path
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        save_path: Optional[str],
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Save path
        self.save_path = to_absolute_path(save_path)

        # z Latent and Latent codes
        self.z_latent = np.load(os.path.join(self.save_path, "output.npz"))["z_latent"]
        self.latent_codes = generator.prepare_latent_codes(
            np.load(os.path.join(self.save_path, "output.npz"))["latent_codes"]
        )

        # Attributes
        self.attributes = pd.read_csv(os.path.join(self.save_path, "attributes.csv"))

    def calculate_metrics(self, latent_name):
        latent_codes = self.z_latent
        if latent_name == "latent_codes":
            latent_codes = self.latent_codes

        dci = DCI(latent_codes, self.attributes, latent_name=latent_name)

        importance_matrix, train_loss, test_loss, models = dci.evaluate()

        scores = {
            "informativeness_train": np.mean(train_loss),
            "informativeness_test": np.mean(test_loss),
        }

        scores = {**scores, **DCI_Test(dci, importance_matrix)}

        return scores

    def dci_metric(self):
        for latent_name in ["z_latent", "latent_codes"]:
            scores = self.calculate_metrics(latent_name)
            self.logger.info(f"{latent_name}: {scores}")
            with open(
                os.path.join(self.save_path, f"{latent_name}_dci_metrics.json"), "w"
            ) as f:
                json.dump(scores, f, indent=4)
