import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dci import DCIMetric
from src.encoder import Encoder
from src.encoder_batch import EncoderBatch
from src.finder import Finder
from src.generator import Generator
from src.manipulator import Manipulator
from src.predictor import Predictor
from src.tuner import Tuner


def generate(cfg: DictConfig) -> None:
    """Generates from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = os.getcwd() if cfg.save else None

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    # Trainer init
    gen = Generator(
        generator=generator,
        num_samples=cfg.num_samples,
        device=device,
        save_path=save_path,
        batch_size=cfg.hparams.batch_size,
    )

    # Launch training process
    gen.generate()


def predict(cfg: DictConfig) -> None:
    """Predicts from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = cfg.load_path

    # Use Hydra's instantiation to initialize directly from the config file
    classifiers: object = instantiate(cfg.classifiers)

    # Trainer init
    predictor = Predictor(
        classifiers=classifiers,
        device=device,
        save_path=save_path,
        batch_size=cfg.hparams.batch_size,
    )

    # Launch training process
    predictor.predict()


def find_directions(cfg: DictConfig) -> None:
    """Finds directions from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = cfg.load_path

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    finder_method: object = instantiate(cfg.finder)

    # Trainer init
    finder = Finder(save_path=save_path, generator=generator, finder=finder_method)

    # Launch training process
    finder.find_directions()


def tune_parameters(cfg: DictConfig) -> None:
    """Tune parameters from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = cfg.load_path
    topks = cfg.topks
    n_images = cfg.n_images
    num_samples = cfg.num_samples
    identity_threshold = cfg.identity_threshold

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    # Trainer init
    tuner = Tuner(
        save_path=save_path,
        generator=generator,
        topks=topks,
        n_images=n_images,
        num_samples=num_samples,
        identity_threshold=identity_threshold,
    )

    # Launch training process
    tuner.tune_parameters()


def manipulate(cfg: DictConfig) -> None:
    """Finds directions from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = cfg.load_path
    latent_codes_path = cfg.latent_codes_path

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    # Trainer init
    manipulator = Manipulator(
        generator=generator,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        topk=cfg.topk,
        device=device,
        save_path=save_path,
        n_images=cfg.n_images,
        extract_shape=cfg.extract_shape,
        latent_codes_path=latent_codes_path,
    )

    # Launch training process
    manipulator.manipulate()


def encode(cfg: DictConfig) -> None:
    """Encodes from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = os.getcwd() if cfg.save else None

    # Use Hydra's instantiation to initialize directly from the config file
    encoder: object = instantiate(cfg.encoder)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    # assert (
    #     encoder.class_name == generator.class_name
    # ), "Classes are different for encoder and generator"

    # Encoder init
    enc = Encoder(
        generator=generator,
        encoder=encoder,
        num_iter=cfg.num_iter,
        device=device,
        save_path=save_path,
        image_path=cfg.image_path,
        batch_size=cfg.hparams.batch_size,
        tune_camera=cfg.tune_camera,
        init_camera={
            "h_mean": cfg.init_camera.h_mean,
            "v_mean": cfg.init_camera.v_mean,
        },
    )

    # Launch training process
    enc.encode()


def encode_batch(cfg: DictConfig) -> None:
    """Encodes from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = os.getcwd() if cfg.save else None

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    encoder: object = instantiate(
        cfg.encoder, generator=generator, data_path=cfg.data_path
    )

    assert (
        encoder.class_name == generator.class_name
    ), "Classes are different for encoder and generator"

    # Encoder init
    enc = EncoderBatch(
        generator=generator,
        encoder=encoder,
        device=device,
        save_path=save_path,
    )

    # Launch training process
    enc.encode()


def dci_metric(cfg: DictConfig) -> None:
    """DCI metric from config
    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    save_path = cfg.load_path

    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)

    # Encoder init
    dci = DCIMetric(save_path=save_path, generator=generator)

    # Launch training process
    dci.dci_metric()


def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config
    Args:
        cfg: Hydra config
    Returns:
        device on which the model will be trained or evaluated
    """
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device
