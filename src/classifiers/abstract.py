from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import numpy as np
import pandas as pd


class Classifiers(AbstractBaseClass):
    """Abstract classifiers
    Args:
        feature_layer: targeted layer to extract features from
    """

    def __init__(self) -> None:
        super(Classifiers, self).__init__()
        pass

    @abstractmethod
    def predict(self, imgs: np.array) -> pd.DataFrame:
        raise NotImplementedError
