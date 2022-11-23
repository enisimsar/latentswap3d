import logging
import os
import sys
from importlib.util import set_loader
from pathlib import Path

import dlib
import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wget
from torch import autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.encoders.abstract import Encoder
from src.generators.abstract import Generator

logger = logging.getLogger(__name__)


class Encoder4Editing(nn.Module):
    def __init__(self, out_dim, num_layers=50, mode="ir_se"):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

        self.styles = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for l in modulelist:
            x = l(x)

        out = self.styles(x).view(x.shape[0], -1)

        return out


class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for i in range(n_mlp - 1):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(style_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)


class E4E_Encoder(Encoder):
    def __init__(
        self,
        device: str,
        generator: Generator,
        data_path: str,
        opts: dict = {},
        class_name: str = "CelebA",
    ) -> None:
        super(E4E_Encoder, self).__init__()

        self.opts = opts
        self.global_step = 0
        self.device = device
        self.class_name = class_name

        self.download_checkpoint()

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = (
                LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
            )
        if self.opts.id_lambda > 0:
            if "ffhq" in class_name.lower() or "celeb" in class_name.lower():
                self.identity_loss = id_loss.IDLoss().to(self.device).eval()
            else:
                self.identity_loss = moco_loss.MocoLoss(opts).to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        out_dim = generator.get_codes(generator.sample_latent(1)).shape[1]

        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(out_dim, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(
                list(self.discriminator.parameters()), lr=opts.w_discriminator_lr
            )
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        # Initialize dataset
        if "ffhq" in class_name.lower() or "celeb" in class_name.lower():
            dataset_paths["ffhq"] = os.path.join(data_path, "ffhq_original")
            dataset_paths["celeba_test"] = os.path.join(data_path, "celeba_test")
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=int(self.opts.workers),
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.opts.test_batch_size,
            shuffle=False,
            num_workers=int(self.opts.test_workers),
            drop_last=True,
        )

        self.encoder = Encoder4Editing(out_dim).to(self.device)

        encoder_ckpt = torch.load(model_paths["ir_se50"])
        self.encoder.load_state_dict(encoder_ckpt, strict=False)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.decoder = generator
        self.decoder.eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

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

            model_paths[n] = str(checkpoint)

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

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        self.requires_grad(self.decoder, False)
        # params += self.decoder.prelearnable_parameters()

        optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        from configs import data_configs

        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception("{} is not a valid dataset_type".format(self.opts.dataset_type))
        print("Loading dataset for {}".format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args["transforms"](self.opts).get_transforms()
        train_dataset = ImagesDataset(
            source_root=dataset_args["train_source_root"],
            target_root=dataset_args["train_target_root"],
            source_transform=transforms_dict["transform_source"],
            target_transform=transforms_dict["transform_gt_train"],
            opts=self.opts,
        )
        test_dataset = ImagesDataset(
            source_root=dataset_args["test_source_root"],
            target_root=dataset_args["test_target_root"],
            source_transform=transforms_dict["transform_source"],
            target_transform=transforms_dict["transform_test"],
            opts=self.opts,
        )
        train_dataset.source_paths = train_dataset.source_paths  # [:400]
        train_dataset.target_paths = train_dataset.target_paths  # [:400]
        test_dataset.source_paths = test_dataset.source_paths[:500]
        test_dataset.target_paths = test_dataset.target_paths[:500]
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def align_face(self, image_path):
        return align_face(filepath=image_path, predictor=self.predictor)

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if "cars" in self.class_name.lower():
            y_hat_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
            y_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y)
            x_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(x)
        else:
            y_hat_resized = self.face_pool(y_hat)
            y_resized = self.face_pool(y)
            x_resized = self.face_pool(x)

        if self.opts.w_discriminator_lambda > 0:
            fake_pred = self.discriminator(latent)
            loss_disc = F.softplus(-fake_pred).mean()
            loss_dict["encoder_discriminator_loss"] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

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

    def forward(self, batch):
        x, y = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        # latent, view_angle = self.encoder.forward(x)
        latent = self.encoder.forward(x)
        if self.opts.start_from_latent_avg:
            latent = latent + self.decoder.mean_latent()
        # self.decoder.h_mean = view_angle[:, 0]
        # self.decoder.v_mean = view_angle[:, 1]
        y_hat = self.decoder.synthesize_inversion(latent, img_size=self.opts.image_size)

        return x, y, y_hat, latent

    def train(self):
        self.encoder.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader)
            ):
                torch.cuda.empty_cache()
                loss_dict = {}
                if self.opts.w_discriminator_lambda > 0:
                    loss_dict = self.train_discriminator(batch)
                x, y, y_hat, latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                # # Validation related
                # val_loss_dict = None
                # if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                #     val_loss_dict = self.validate()
                #     if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                #         self.best_val_loss = val_loss_dict['loss']
                #         self.checkpoint_me(val_loss_dict, is_best=True)

                # if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                #     if val_loss_dict is not None:
                #         self.checkpoint_me(val_loss_dict, is_best=False)
                #     else:
                #         self.checkpoint_me(loss_dict, is_best=False)

                if batch_idx % 200 == 0 and batch_idx < 1601:
                    img = self.decoder.synthesize(latent)[0]
                    print(self.decoder.h_mean, self.decoder.v_mean)
                    # print(list(self.encoder.styles.children())[-1].weight.grad)
                    torch.save(self.encoder, "encoder.pt")
                    from PIL import Image

                    os.makedirs("remove_images", exist_ok=True)

                    Image.fromarray((img * 255).astype(np.uint8)).save(
                        f"remove_images/{self.global_step}_{batch_idx}.png"
                    )
                    print(loss_dict)

                if self.global_step == self.opts.max_steps:
                    print("OMG, finished training!")
                    break

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict["d_real_loss"] = float(real_loss)
        loss_dict["d_fake_loss"] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):
        loss_dict = {}
        x, _ = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict["discriminator_loss"] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = (
                self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            )
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict["discriminator_r1_loss"] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict["discriminator_loss"] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = self.decoder.sample_latent(self.opts.batch_size)
        real_w = self.decoder.get_codes(sample_z)
        # fake_w, _ = self.encoder(x)
        fake_w = self.encoder(x)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.decoder.mean_latent().repeat(fake_w.shape[0], 1)
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        return real_w, fake_w
