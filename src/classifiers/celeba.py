import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from PIL import Image
from tqdm import tqdm

from src.classifiers.abstract import Classifiers

module_path = Path(__file__).parent / "stylegan2"
sys.path.insert(1, str(module_path.resolve()))

import dnnlib.tflib as tflib


class CelebAClassifiers(Classifiers):
    def __init__(
        self,
        device: str,
        classifiers_path: str,
    ) -> None:
        super(CelebAClassifiers, self).__init__()

        self.device = device
        self.classifiers_path = to_absolute_path(classifiers_path)

        self.IMAGE_SHAPE = (256, 256)

    def find_classifiers(self) -> None:
        """Finds the classifiers"""
        names_tmp = os.listdir(self.classifiers_path)
        names = []
        for name in names_tmp:
            if "celebahq-classifier" in name:
                names.append(name)
        names.sort()
        self.classifiers = names

    def _convert_images_from_uint8(self, images, drange=[-1, 1], nhwc_to_nchw=False):
        """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
        Can be used as an input transformation for Network.run().
        """
        if nhwc_to_nchw:
            imgs_roll = np.rollaxis(images, 3, 1)
        return imgs_roll / 255 * (drange[1] - drange[0]) + drange[0]

    def predict(self, imgs: np.array, batch_size: int) -> pd.DataFrame:
        """Predicts from classifiers
        Args:
            imgs: images to predict
        """
        # Initialize
        self.find_classifiers()
        results = {}

        tflib.init_tf()
        # Predict
        for name in tqdm(self.classifiers):
            tmp = os.path.join(self.classifiers_path, name)
            with open(tmp, "rb") as f:
                classifier = pickle.load(f)

            logits = np.zeros(len(imgs))
            for i in range(int(imgs.shape[0] / batch_size)):
                tmp_imgs = imgs[(i * batch_size) : ((i + 1) * batch_size)]

                tmp_images2 = []
                for img in tmp_imgs:
                    if img.max() < 1.01:
                        img = (img * 255).astype("uint8")
                    images = Image.fromarray(img).resize(self.IMAGE_SHAPE)
                    images = np.array(images)
                    tmp_images2.append(images)

                tmp_images2 = np.array(tmp_images2)
                tmp_imgs = self._convert_images_from_uint8(
                    tmp_images2, drange=[-1, 1], nhwc_to_nchw=True
                )
                tmp = classifier.run(tmp_imgs, None)

                tmp1 = tmp.reshape(-1)
                logits[(i * batch_size) : ((i + 1) * batch_size)] = tmp1

            tmp1 = name[20:-4]
            results[tmp1] = logits

        # Save
        df = pd.DataFrame(results)
        return df
