import logging
import os
from typing import Optional

import numpy as np
import torch
from hydra.utils import to_absolute_path


class Predictor:
    """Model predictor
    Args:
        classifiers: pretrained classifiers
        num_samples: number of samples
        device: device on which to evaluate model
        save_path: save path
    """

    def __init__(
        self,
        classifiers: object,
        device: torch.device,
        save_path: Optional[str],
        batch_size: int,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Classifiers
        self.classifiers = classifiers

        # Batch size & Number of samples
        self.batch_size = batch_size

        # Save path
        self.save_path = to_absolute_path(save_path)

        # Images
        self.imgs = np.load(os.path.join(self.save_path, "output.npz"))["imgs"]

    def predict(self):
        """Predicts images"""

        results = self.classifiers.predict(self.imgs, self.batch_size)

        os.makedirs(self.save_path, exist_ok=True)

        results.to_csv(os.path.join(self.save_path, "attributes.csv"), index=False)
