import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


class SelectKBestFinder(object):
    """Lasso Finder"""

    def __init__(self, topk):
        self.topk = topk

    def init_model(self):
        self.feature_importance = None
        self.selectkbest = SelectKBest(chi2, k=self.topk)

    def fit(self, X, y):
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        self.selectkbest.fit(X, y)

        self.feature_importance = (
            self.selectkbest.scores_
        )  # self.selectkbest.get_support(True).tolist() + [i for i in range(X.shape[1]) if i not in self.selectkbest.get_support(True)]

    def predict(self, X):
        if self.feature_importance is None:
            raise ValueError("Model is not fitted yet.")

        return (np.random.randn(len(X)) > 0.5).astype(int)

    def feature_importances_(self, X=None):
        return self.feature_importance
