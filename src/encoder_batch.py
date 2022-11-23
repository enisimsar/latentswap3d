import logging
import os
from typing import Optional

import numpy as np
import torch
import tqdm
from hydra.utils import to_absolute_path
from PIL import Image
from torchvision.utils import save_image


class EncoderBatch:
    """Model encoder
    Args:
        generator: pretrained generator
        num_samples: number of samples
        device: device on which to evaluate model
        save_path: save path
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        encoder: object,
        device: torch.device,
        save_path: Optional[str],
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Generator & Encoder
        self.encoder = encoder
        self.generator = generator

        # Save path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def encode(self):
        """Encodes images"""
        self.encoder.train()
