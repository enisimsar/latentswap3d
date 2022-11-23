import logging
import os
import sys
from pathlib import Path

import dlib
import gdown
import numpy as np
import torch
import torch.nn as nn
import wget
from torchvision import transforms

from src.criteria import id_loss, moco_loss
from src.criteria.lpips.lpips import LPIPS
from src.encoders.abstract import Encoder
from src.utils.alignment import align_face

logger = logging.getLogger(__name__)


class VanillaEncoder(Encoder):
    def __init__(
        self,
        device: str,
        opts: dict = {},
        class_name: str = "CelebA",
    ) -> None:
        super(VanillaEncoder, self).__init__()

        self.opts = opts
        self.device = device
        self.class_name = class_name
        self.model_paths = {}

        self.download_checkpoint()

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = (
                LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
            )
        if self.opts.id_lambda > 0:
            if "ffhq" in class_name.lower() or "celeb" in class_name.lower():
                self.identity_loss = (
                    id_loss.IDLoss(self.model_paths["ir_se50"]).to(self.device).eval()
                )
            else:
                self.identity_loss = (
                    moco_loss.MocoLoss(self.model_paths["moco"]).to(self.device).eval()
                )
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.transforms = transforms.Compose(
            [
                # transforms.Resize(320),
                transforms.Resize(304),
                transforms.CenterCrop(256),
                transforms.Resize(
                    (self.opts.image_size, self.opts.image_size), interpolation=0
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self):
        models = {
            (
                "ir_se50",
                "model_ir_se50.pth",
            ): "https://drive.google.com/uc?export=download&id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
            # "shape_predictor_68_face_landmarks.dat": ,
            (
                "moco",
                "moco_v2_800ep_pretrain.pth",
            ): "https://drive.google.com/uc?export=download&id=18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe",
        }

        checkpoint_root = Path(__file__).parent / "checkpoints"
        os.makedirs(checkpoint_root, exist_ok=True)

        for (n, m), url in models.items():
            checkpoint = Path(checkpoint_root) / m
            if not checkpoint.is_file():
                logger.info(f"Downloading {m}")
                gdown.download(url, str(checkpoint), quiet=False)

            self.model_paths[n] = str(checkpoint)

        if "ffhq" in self.class_name.lower() or "celeb" in self.class_name.lower():
            m = "shape_predictor_68_face_landmarks.dat"
            checkpoint = Path(checkpoint_root) / m
            if not checkpoint.is_file():
                logger.info(f"Downloading {m}")
                wget.download(
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                    str(checkpoint) + ".bz2",
                )
                import bz2

                # decompress data
                with bz2.open(str(checkpoint) + ".bz2", "rb") as f:
                    uncompressed_content = f.read()

                # store decompressed file
                with open(str(checkpoint), "wb") as f:
                    f.write(uncompressed_content)
                    f.close()

            self.predictor = dlib.shape_predictor(str(checkpoint))

    def init_optimizers(self, parameters):
        self.optimizer = torch.optim.Adam(
            parameters, lr=self.opts.lr, betas=(0.5, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.opts.lr_step, gamma=0.75
        )

    def align_face(self, image_path):
        return align_face(filepath=image_path, predictor=self.predictor)

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if y_hat.shape[0] == 1 and x.shape[0] == 2:
            x = x[:1]
            y = y[1:]

        if "cars" in self.class_name.lower():
            y_hat_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
            y_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y)
            x_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(x)
        else:
            y_hat_resized = self.face_pool(y_hat)
            y_resized = self.face_pool(y)
            x_resized = self.face_pool(x)

        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.identity_loss(
                y_hat_resized, y_resized, x_resized
            )
            loss_dict["loss_id"] = float(loss_id)
            loss_dict["id_improve"] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = self.mse_loss(y_hat, y)
            loss_dict["loss_l2"] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat_resized, y_resized)
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(
                y_hat_resized, y_resized, x_resized
            )
            loss_dict["loss_moco"] = float(loss_moco)
            loss_dict["id_improve"] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict["loss"] = float(loss)
        return loss, loss_dict, id_logs
