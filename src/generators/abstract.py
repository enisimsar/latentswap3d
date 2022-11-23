from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import numpy as np
import torch


class Generator(AbstractBaseClass, torch.nn.Module):
    """Abstract generator"""

    def __init__(self) -> None:
        super(Generator, self).__init__()
        pass

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, code: torch.Tensor, angle: float = None) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def synthesize_inversion(self, codes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def mean_latent(self) -> torch.Tensor:
        raise NotImplementedError

    def prepare_latent_codes(self, codes) -> np.array:
        return codes

    def manipulated_indices(self, indices) -> np.array:
        return indices
