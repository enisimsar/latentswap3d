import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC


class SVMFinder(object):
    """Lasso Finder"""

    def __init__(self):
        pass

    def init_model(self):
        self.feature_importance = None
        self.lasso = SVC(kernel="linear", C=0.025)

    def fit(self, X, y):
        self.lasso.fit(X, y)

        # result = permutation_importance(self.lasso, X, y, n_repeats=10, random_state=0)

        self.feature_importance = np.abs(
            self.lasso.coef_.ravel()
        )  # result.importances_mean

    def predict(self, X):
        if self.feature_importance is None:
            raise ValueError("Model is not fitted yet.")

        return (np.random.randn(len(X)) > 0.5).astype(int)

    def feature_importances_(self, X=None):
        return self.feature_importance
