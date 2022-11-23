import logging
import os
from copy import deepcopy as dc
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from hydra.utils import to_absolute_path
from PIL import Image
from torchvision.utils import save_image


class Encoder:
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
        num_iter: int,
        device: torch.device,
        save_path: Optional[str],
        image_path: Optional[str],
        tune_camera: bool,
        init_camera: dict,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Generator & Encoder
        self.encoder = encoder
        self.generator = generator

        # Batch size & Number of iterations
        self.num_iter = num_iter
        self.tune_camera = tune_camera
        self.init_camera = init_camera

        # Save and image path
        self.image_path = to_absolute_path(image_path)
        suffix = image_path.split("/")[-1]
        if os.path.isfile(self.image_path):
            suffix = image_path.split("/")[-1].split(".")[0]

        tune = "with_camera" if self.tune_camera else "without_camera"
        suffix = f"{suffix}_{tune}"
        self.save_path = os.path.join(save_path, suffix)
        os.makedirs(self.save_path, exist_ok=True)

    def train_loop(self, iteration, mean_codes, gt_image, offsets):
        noise_w = (
            0.03
            * torch.randn_like(mean_codes)
            * (self.num_iter - iteration)
            / self.num_iter
        )

        tensor_img = self.generator.synthesize_inversion(
            mean_codes + noise_w + offsets, img_size=self.encoder.opts.image_size
        )
        loss, loss_dict, _ = self.encoder.calc_loss(
            gt_image, gt_image, tensor_img, mean_codes + offsets
        )

        del tensor_img
        torch.cuda.empty_cache()  # Clear GPU memory

        return loss, loss_dict

    def iterate_views(self, iteration, mean_codes, gt_image, offsets):
        total_loss = 0
        total_dict = []
        for idx, img in enumerate(gt_image):
            img = torch.unsqueeze(img, 0)
            self.generator.h_mean = self.angle_views[idx][0]
            self.generator.v_mean = self.angle_views[idx][1]
            loss, loss_dict = self.train_loop(iteration, mean_codes, img, offsets)
            total_loss += loss
            total_dict.append(loss_dict)

        return total_loss / len(gt_image), dict(pd.DataFrame(total_dict).mean())

    def optimize(self, mean_codes, gt_image, offsets, num_iteration):
        parameters = []
        for angle in self.angle_views:
            parameters.append(angle[0])
            parameters.append(angle[1])
        camera_optimizer = torch.optim.Adam(
            parameters,
            lr=self.encoder.opts.lr,
            betas=(0.5, 0.999),
        )
        camera_scheduler = torch.optim.lr_scheduler.StepLR(
            camera_optimizer, step_size=self.encoder.opts.lr_step // 2, gamma=0.75
        )

        parameters = [offsets]

        self.encoder.init_optimizers(parameters)

        # Loop
        for i in tqdm.trange(num_iteration):
            if self.tune_camera and i % 100 == 0 and i < 200:
                for j in range(50):
                    camera_optimizer.zero_grad()

                    total_loss, _ = self.iterate_views(i, mean_codes, gt_image, offsets)

                    total_loss.backward()
                    camera_optimizer.step()
                    camera_scheduler.step()
                    if j % 50 == 0:
                        for view_idx, (h_mean, v_mean) in enumerate(self.angle_views):
                            self.logger.info(
                                f"View #{view_idx} --> h_mean: {h_mean.detach().cpu().numpy()[0]:.2f}, v_mean: {v_mean.detach().cpu().numpy()[0]:.2f}"
                            )
                        self.save_alignment_image(j + i, mean_codes + offsets)
                    del total_loss
                    torch.cuda.empty_cache()  # Clear GPU memory

            self.encoder.optimizer.zero_grad()

            total_loss, total_dict = self.iterate_views(
                i, mean_codes, gt_image, offsets
            )

            total_loss.backward()
            self.encoder.optimizer.step()
            self.encoder.scheduler.step()

            del total_loss
            torch.cuda.empty_cache()  # Clear GPU memory

            if i % self.encoder.opts.lr_step == 0:
                self.logger.info(total_dict)
                self.save_image(i, mean_codes + offsets)

    def save_image(self, name, codes):
        imgs = []
        for angle in [-0.3, -0.15, 0, 0.15, 0.3]:
            img = (self.generator.synthesize(codes, angle, 0) * 255).astype(np.uint8)[0]
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        Image.fromarray(imgs).save(os.path.join(self.save_path, f"{name}.jpg"))

    def save_alignment_image(self, name, codes):
        imgs = []
        for h_mean, v_mean in self.angle_views:
            img = (
                self.generator.synthesize(
                    codes,
                    float(h_mean.detach().cpu().numpy()[0]),
                    float(v_mean.detach().cpu().numpy()[0]),
                )
                * 255
            ).astype(np.uint8)[0]
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        Image.fromarray(imgs).save(
            os.path.join(self.save_path, f"alignment_{name}.jpg")
        )

    def encode(self):
        """Encodes images"""
        model_copy = dc(self.generator.model).cpu()

        mean_codes = dc(self.generator.mean_latent())
        mean_codes = (
            torch.cat(mean_codes, axis=1) if type(mean_codes) == list else mean_codes
        )

        imgs_paths = None
        if os.path.isfile(self.image_path):
            imgs_paths = [self.image_path]
        else:
            paths = glob(os.path.join(self.image_path, "*.png"))
            from random import shuffle

            shuffle(paths)
            imgs_paths = paths[:2]

        images = []
        for img_path in imgs_paths:
            Image.fromarray(np.array(Image.open(img_path))[:, :, :3]).save(img_path)

            if (
                "ffhq" in self.encoder.class_name.lower()
                or "celeb" in self.encoder.class_name.lower()
            ):
                gt_image = self.encoder.align_face(img_path)
            else:
                gt_image = Image.open(img_path).convert("RGB")

            images.append(self.encoder.transforms(gt_image).to(self.device))

        gt_image = torch.stack(images)
        save_image(gt_image, os.path.join(self.save_path, "gt.jpg"), normalize=True)

        self.angle_views = []
        for _ in range(len(gt_image)):
            h_mean = torch.zeros(1).to(self.device) + self.init_camera["h_mean"]
            h_mean.requires_grad_()

            v_mean = torch.zeros(1).to(self.device) + self.init_camera["v_mean"]
            v_mean.requires_grad_()

            self.angle_views.append((h_mean, v_mean))

        offsets = torch.zeros_like(mean_codes)
        offsets.requires_grad_()

        self.optimize(mean_codes, gt_image, offsets, num_iteration=self.num_iter)

        for view_idx, (h_mean, v_mean) in enumerate(self.angle_views):
            self.logger.info(
                f"View #{view_idx}: h_mean: {h_mean.detach().cpu().numpy()[0]:.2f}, v_mean: {v_mean.detach().cpu().numpy()[0]:.2f}"
            )

        self.save_image("final", mean_codes + offsets)

        np.savez_compressed(
            os.path.join(self.save_path, "latent_codes"),
            latent_codes=(mean_codes + offsets).detach().cpu().numpy(),
        )
