import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


class NIFinder(object):
    """Random Forest Finder"""

    def __init__(self, nb_runs):
        self.nb_runs = nb_runs

    def init_model(self):
        pass

    def get_feature_importances(self, data, shuffle, seed=None):
        # Gather real features
        train_features = [f for f in data if f not in ["TARGET"]]
        # Go over fold and keep track of CV score (train and valid) and feature importances

        # Shuffle target if required
        y = data["TARGET"].copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = data["TARGET"].copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
        lgb_params = {
            "objective": "binary",
            "boosting_type": "rf",
            "subsample": 0.623,
            "colsample_bytree": 0.7,
            "num_leaves": 127,
            "max_depth": 8,
            "seed": seed,
            "bagging_freq": 1,
            "n_jobs": 4,
        }

        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type="gain")
        imp_df["importance_split"] = clf.feature_importance(importance_type="split")
        imp_df["trn_score"] = roc_auc_score(y, clf.predict(data[train_features]))

        return imp_df

    def fit(self, X, y):
        data = pd.DataFrame(X, columns=list(map(str, range(X.shape[1]))))
        data["TARGET"] = y

        actual_imp_df = self.get_feature_importances(data=data, shuffle=False)

        null_imp_df = pd.DataFrame()
        for i in range(self.nb_runs):
            # Get current run importances
            imp_df = self.get_feature_importances(data=data, shuffle=True)
            imp_df["run"] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

        feature_scores = []
        for _f in actual_imp_df["feature"].unique():
            f_null_imps_gain = null_imp_df.loc[
                null_imp_df["feature"] == _f, "importance_gain"
            ].values
            f_act_imps_gain = actual_imp_df.loc[
                actual_imp_df["feature"] == _f, "importance_gain"
            ].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75))
            )  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[
                null_imp_df["feature"] == _f, "importance_split"
            ].values
            f_act_imps_split = actual_imp_df.loc[
                actual_imp_df["feature"] == _f, "importance_split"
            ].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75))
            )  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(
            feature_scores, columns=["feature", "split_score", "gain_score"]
        )

        correlation_scores = []
        for _f in actual_imp_df["feature"].unique():
            f_null_imps = null_imp_df.loc[
                null_imp_df["feature"] == _f, "importance_gain"
            ].values
            f_act_imps = actual_imp_df.loc[
                actual_imp_df["feature"] == _f, "importance_gain"
            ].values
            gain_score = (
                100
                * (f_null_imps < np.percentile(f_act_imps, 25)).sum()
                / f_null_imps.size
            )
            f_null_imps = null_imp_df.loc[
                null_imp_df["feature"] == _f, "importance_split"
            ].values
            f_act_imps = actual_imp_df.loc[
                actual_imp_df["feature"] == _f, "importance_split"
            ].values
            split_score = (
                100
                * (f_null_imps < np.percentile(f_act_imps, 25)).sum()
                / f_null_imps.size
            )
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(
            correlation_scores, columns=["feature", "split_score", "gain_score"]
        )

        self.feature_importance = (
            corr_scores_df.sort_values("split_score", ascending=False)
            .feature.values.astype(int)
            .tolist()
        )

    def predict(self, X):
        if self.feature_importance is None:
            raise ValueError("Model is not fitted yet.")

        return (np.random.randn(len(X)) > 0.5).astype(int)

    def feature_importances_(self, X=None):
        return self.feature_importance
