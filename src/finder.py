import json
import logging
import math
import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Finder:
    """Model Finder
    Args:
        generator: pretrained generator
        save_path: save path
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        finder: object,
        save_path: Optional[str],
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Generator
        self.generator = generator

        # finder
        self.finder = finder

        # Save path
        self.save_path = to_absolute_path(save_path)

        # Latent codes
        self.latent_codes = generator.prepare_latent_codes(
            np.load(os.path.join(self.save_path, "output.npz"))["latent_codes"]
        )

        # Attributes
        self.attributes = pd.read_csv(os.path.join(self.save_path, "attributes.csv"))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def calculate_scores(self, target, pred):
        results = {}
        for metric in [f1_score, recall_score, precision_score]:
            for average in ["micro", "macro", "weighted"]:
                results[f"{metric.__name__}_{average}"] = float(
                    f"{metric(target, pred, average=average):.2f}"
                )
        results["accuracy"] = float(f"{accuracy_score(target, pred):.2f}")
        results["confusion_matrix"] = confusion_matrix(target, pred).tolist()
        results["support"] = (
            np.array([int((target == 0).sum()), int((target == 1).sum())])
            .astype(np.uint)
            .tolist()
        )
        results["is_valid"] = bool(
            ((target == 0).sum() != 0) and ((target == 1).sum() != 0)
        )
        return results

    def find_directions(self):
        """Finds directions"""

        results = {}
        metrics = {}
        for attr in tqdm(self.attributes.columns):
            try:
                self.finder.init_model()

                df = pd.DataFrame(self.latent_codes)
                df["target"] = (
                    self.attributes[attr]
                    .apply(lambda x: int(self.sigmoid(x) > 0.5))
                    .values
                )

                # negative_target = df[df.target == 0].copy()
                # positive_target = df[df.target == 1].copy()

                # if len(negative_target) > len(positive_target):
                #     df = pd.concat([positive_target, negative_target.sample(len(positive_target), random_state=1881)])
                # else: # len(negative_target) > len(positive_target)
                #     df = pd.concat([negative_target, positive_target.sample(len(negative_target), random_state=1881)])

                X = df.values[:, :-1]
                y = df.target.values

                X_train, X_test, ind_train, ind_test = train_test_split(
                    X, range(len(X)), test_size=0.25, random_state=42, stratify=y
                )
                y_train = y[ind_train]
                y_test = y[ind_test]

                self.finder.fit(X_train, y_train)

                metrics[attr] = {}
                y_pred = self.finder.predict(X_train) > 0.5
                metrics[attr]["train"] = self.calculate_scores(y_train, y_pred)

                y_pred = self.finder.predict(X_test) > 0.5
                metrics[attr]["test"] = self.calculate_scores(y_test, y_pred)

                metrics[attr]["is_valid"] = (
                    metrics[attr]["train"]["is_valid"]
                    and metrics[attr]["test"]["is_valid"]
                )

                feature_scores = pd.Series(
                    self.finder.feature_importances_(X_test),
                    index=range(self.latent_codes.shape[1]),
                ).sort_values(ascending=False)

                results[attr] = list(feature_scores.index)
            except Exception as e:
                print(e)

        json.dump(
            results, open(os.path.join(self.save_path, "results.json"), "w"), indent=4
        )
        json.dump(
            metrics, open(os.path.join(self.save_path, "metrics.json"), "w"), indent=4
        )
