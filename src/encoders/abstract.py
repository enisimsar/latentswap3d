from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import torch


class Encoder(AbstractBaseClass):
    """Abstract encoder"""

    def __init__(self) -> None:
        super(Encoder, self).__init__()
        pass
