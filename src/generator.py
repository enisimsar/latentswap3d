import logging
import os
from typing import Optional

import numpy as np
import torch
import tqdm


class Generator:
    """Model generator
    Args:
        generator: pretrained generator
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
        batch_size: int,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Generator
        self.generator = generator

        # Batch size & Number of samples
        self.batch_size = batch_size
        self.num_samples = num_samples

        # Save path
        self.save_path = save_path

    def generate(self):
        """Generates images"""

        # Progress bar
        pbar = tqdm.tqdm(total=self.num_samples // self.batch_size)
        pbar.set_description("Generating... ")

        z_latent = []
        imgs = []
        latent_codes = []
        # Loop
        for seed in range(0, self.num_samples, self.batch_size):
            z = self.generator.sample_latent(self.batch_size, seed)
            codes = self.generator.get_codes(z)

            img = self.generator.synthesize(codes)

            for im in img:
                imgs.append(im)

            for code in codes:
                latent_codes.append(code.detach().cpu().numpy())

            # TODO: put more conditions here
            if type(z) == dict:
                z = z["z_app_obj"]  # giraffe generator

            for z_l in z:
                z_latent.append(z_l.detach().cpu().numpy())

            # Update progress bar
            pbar.update()

        pbar.close()

        imgs = np.stack(imgs)
        latent_codes = np.stack(latent_codes)
        z_latent = np.stack(z_latent)

        os.makedirs(self.save_path, exist_ok=True)

        np.savez_compressed(
            os.path.join(self.save_path, "output"),
            imgs=imgs,
            z_latent=z_latent,
            latent_codes=latent_codes,
        )
