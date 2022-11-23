import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from hydra.utils import to_absolute_path
from PIL import Image
from tqdm import tqdm

from src.classifiers.abstract import Classifiers

module_path = Path(__file__).parent / "stylegan2"
sys.path.insert(1, str(module_path.resolve()))

import dnnlib.tflib as tflib

transform_valid = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class CatsClassifiers(Classifiers):
    def __init__(
        self,
        device: str,
        classifiers_path: str,
    ) -> None:
        super(CatsClassifiers, self).__init__()

        self.device = device
        self.classifiers_path = to_absolute_path(classifiers_path)

        self.IMAGE_SHAPE = (256, 256)

    def find_classifiers(self) -> None:
        """Finds the classifiers"""
        names_tmp = os.listdir(self.classifiers_path)
        names = []
        for name in names_tmp:
            if "cats-classifier" in name:
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

    def _load_model(self, checkpoint):
        # initialize model
        model = torchvision.models.resnet50(pretrained=True).to(self.device)

        # freeze the backbone
        for parameter in model.parameters():
            parameter.requires_grad = False

        class ModelHead(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, n_classes):
                super(ModelHead, self).__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
                self.relu2 = torch.nn.ReLU()
                self.fc3 = torch.nn.Linear(hidden_dim // 2, n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        model.fc = ModelHead(2048, 1024, 2)
        model.fc.to(self.device)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        return model

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
            classifier = self._load_model(tmp)

            logits = np.zeros(len(imgs))
            for i in range(int(imgs.shape[0] / batch_size)):
                tmp_imgs = imgs[(i * batch_size) : ((i + 1) * batch_size)]

                test_data = torch.stack(
                    [transform_valid(((img * 255).astype("uint8"))) for img in tmp_imgs]
                ).to(self.device)
                with torch.no_grad():
                    test_preds = classifier(test_data)

                logits[(i * batch_size) : ((i + 1) * batch_size)] = (
                    test_preds[:, 1].detach().cpu().numpy()
                )

            tmp1 = name[len("cats-classifier-") : -4]
            results[tmp1] = logits

        # Save
        df = pd.DataFrame(results)
        return df
