import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class RFFinder(object):
    """Random Forest Finder"""

    def __init__(self, is_shap=False):
        self.is_shap = is_shap

    def init_model(self):
        param_grid = {
            "bootstrap": [True],
            "max_depth": [10, 20, 30],
            "max_features": [2, 3],
            "min_samples_leaf": [3, 4, 5],
            "min_samples_split": [8, 10, 12],
            "n_estimators": [20, 50, 100],
        }
        # Create a based model
        rf = RandomForestRegressor()
        # Instantiate the grid search model
        self.grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        self.rf = None

    def fit(self, X, y):
        self.grid_search.fit(X, y)  # (X_test, y_test)
        self.rf = self.grid_search.best_estimator_

    def predict(self, X):
        if self.rf is None:
            raise ValueError("Model is not fitted yet.")

        return self.rf.predict(X) > 0.5

    def feature_importances_(self, X=None):
        if self.is_shap:
            explainer = shap.TreeExplainer(self.rf)
            shap_values = explainer(X)
            return np.abs(shap_values.values).mean(0)
        return self.rf.feature_importances_
