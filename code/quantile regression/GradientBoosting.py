#!/usr/bin/env python
# coding: utf-8

# # Quantile Regression with Gradient Boosting Models

# ## Imports

import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin


from sklearn.ensemble import GradientBoostingRegressor


from lightgbm import LGBMRegressor


from catboost import CatBoostRegressor


# ## Scikit-Learn Gradient Boosting Regressor


class GradientBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for scikit-learn GradientBoostingRegressor which provides functionality to jointly predict several quantiles.
    
    An independent model is used for each quantile. However, with cascade=True, the models can be partially linked
    by using the already predicted lower quantile for predicting the next higher quantile. Parameters for all models
    as well as for specific models can be set using base_params and quantile_params.
    
    Parameters
    ----------
    quantiles : iterable
        Quantiles to predict, should be in ascending order.
    base_params : dict, optional
        Dictionary with parameter mappings to apply to all models.
    quantile_params: dict of dicts, optional
        Dictionary with quantiles as keys and dictionaries with parameter mappings
        for corresponding specific models as values.
    cascade: bool, optional
        If True, the predicted lower quantile is used by the model for next higher quantile as well. Default is False.
    """

    def __init__(
        self, quantiles, base_params=None, quantile_params=None, cascade=False
    ):
        if base_params is None:
            base_params = dict()
        if quantile_params is None:
            quantile_params = dict()
        self.models_ = dict(
            zip(
                quantiles,
                [
                    GradientBoostingRegressor(**base_params, loss="quantile", alpha=q)
                    for q in quantiles
                ],
            )
        )
        for q, params in quantile_params.items():
            self.models_[q] = self.models_[q].set_params(**params)

        self.quantiles = quantiles
        self.base_params = base_params
        self.quantile_params = quantile_params
        self.cascade = cascade

    def fit(self, X, y):
        if not self.cascade:
            X = np.array(X)
            for q, m in self.models_.items():
                m.fit(X, y)
        else:
            X_ = np.array(X)
            for q, m in self.models_.items():
                m.fit(X_, y)
                y_ = m.predict(X_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))

        return self

    def predict(self, X):
        if not self.cascade:
            X = np.array(X)
            predictions = np.array(
                [m.predict(X) for m in self.models_.values()]
            ).transpose()
        else:
            predictions = list()
            X_ = np.array(X)
            for m in self.models_.values():
                y_ = m.predict(X_)
                predictions.append(y_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))
            predictions = np.array(predictions).transpose()
        return predictions

    def get_params(self, deep=True):
        return dict(
            quantiles=self.quantiles,
            base_params=self.base_params,
            quantile_params=self.quantile_params,
            cascade=self.cascade,
        )


# ## LightGBM Regressor


class LightGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for LightGBMRegressor which provides functionality to jointly predict several quantiles.
    
    An independent model is used for each quantile. However, with cascade=True, the models can be partially linked
    by using the already predicted lower quantile for predicting the next higher quantile. Parameters for all models
    as well as for specific models can be set using base_params and quantile_params.
    
    Parameters
    ----------
    quantiles : iterable
        Quantiles to predict, should be in ascending order.
    base_params : dict, optional
        Dictionary with parameter mappings to apply to all models.
    quantile_params: dict of dicts, optional
        Dictionary with quantiles as keys and dictionaries with parameter mappings
        for corresponding specific models as values.
    cascade: bool, optional
        If True, the predicted lower quantile is used by the model for next higher quantile as well. Default is False.
    """

    def __init__(
        self, quantiles, base_params=dict(), quantile_params=dict(), cascade=False
    ):
        if base_params is None:
            base_params = dict()
        if quantile_params is None:
            quantile_params = dict()
        self.models_ = dict(
            zip(
                quantiles,
                [
                    LGBMRegressor(**base_params, objective="quantile", alpha=q)
                    for q in quantiles
                ],
            )
        )
        for q, params in quantile_params.items():
            self.models_[q] = self.models_[q].set_params(**params)

        self.quantiles = quantiles
        self.base_params = base_params
        self.quantile_params = quantile_params
        self.cascade = cascade

    def fit(self, X, y):
        if not self.cascade:
            X = np.array(X)
            for q, m in self.models_.items():
                m.fit(X, y)
        else:
            X_ = np.array(X)
            for q, m in self.models_.items():
                m.fit(X_, y)
                y_ = m.predict(X_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))

        return self

    def predict(self, X):
        if not self.cascade:
            X = np.array(X)
            predictions = np.array(
                [m.predict(X) for m in self.models_.values()]
            ).transpose()
        else:
            predictions = list()
            X_ = np.array(X)
            for m in self.models_.values():
                y_ = m.predict(X_)
                predictions.append(y_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))
            predictions = np.array(predictions).transpose()
        return predictions

    def get_params(self, deep=True):
        return dict(
            quantiles=self.quantiles,
            base_params=self.base_params,
            quantile_params=self.quantile_params,
            cascade=self.cascade,
        )


# ## CatBoost Regressor


class CatBoostQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for CatBoostRegressor which provides functionality to jointly predict several quantiles.
    
    An independent model is used for each quantile. However, with cascade=True, the models can be partially linked
    by using the already predicted lower quantile for predicting the next higher quantile. Parameters for all models
    as well as for specific models can be set using base_params and quantile_params.
    
    Parameters
    ----------
    quantiles : iterable
        Quantiles to predict, should be in ascending order.
    base_params : dict, optional
        Dictionary with parameter mappings to apply to all models.
    quantile_params: dict of dicts, optional
        Dictionary with quantiles as keys and dictionaries with parameter mappings
        for corresponding specific models as values.
    cascade: bool, optional
        If True, the predicted lower quantile is used by the model for next higher quantile as well. Default is False.
    """

    def __init__(
        self, quantiles, base_params=dict(), quantile_params=dict(), cascade=False
    ):
        if base_params is None:
            base_params = dict()
        if quantile_params is None:
            quantile_params = dict()
        self.models_ = dict(
            zip(
                quantiles,
                [
                    CatBoostRegressor(
                        **base_params, loss_function=f"Quantile:alpha={q}"
                    )
                    for q in quantiles
                ],
            )
        )
        for q, params in quantile_params.items():
            self.models_[q] = self.models_[q].set_params(**params)

        self.quantiles = quantiles
        self.base_params = base_params
        self.quantile_params = quantile_params
        self.cascade = cascade

    def fit(self, X, y):
        if not self.cascade:
            X = np.array(X)
            for q, m in self.models_.items():
                m.fit(X, y)
        else:
            X_ = np.array(X)
            for q, m in self.models_.items():
                m.fit(X_, y)
                y_ = m.predict(X_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))

        return self

    def predict(self, X):
        if not self.cascade:
            X = np.array(X)
            predictions = np.array(
                [m.predict(X) for m in self.models_.values()]
            ).transpose()
        else:
            predictions = list()
            X_ = np.array(X)
            for m in self.models_.values():
                y_ = m.predict(X_)
                predictions.append(y_)
                X_ = np.hstack((X_, y_.reshape(-1, 1)))
            predictions = np.array(predictions).transpose()
        return predictions

    def get_params(self, deep=True):
        return dict(
            quantiles=self.quantiles,
            base_params=self.base_params,
            quantile_params=self.quantile_params,
            cascade=self.cascade,
        )


# ## Tests

if __name__ == "__main__":
    import pandas as pd

    noro_ts = pd.read_pickle("../../data/processed/norovirus-ts.pl")


if __name__ == "__main__":
    input_window = 5
    forecast_horizon = 1
    ts_shifted = pd.concat(
        [
            noro_ts[["Count"]].shift(i).rename(columns={"Count": f"Count_t-{i}"})
            for i in range(input_window)
        ],
        axis=1,
    )
    ts_shifted["target"] = ts_shifted["Count_t-0"].shift(-forecast_horizon)
    ts_shifted = ts_shifted.dropna()
    X = ts_shifted.drop("target", axis=1)
    y = ts_shifted["target"]


# ### Scikit-learn GradientBoosting

if __name__ == "__main__":
    gbqr = GradientBoostingQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(n_estimators=102),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=False,
    )
    gbqr = gbqr.fit(X, y)
    y_hat = gbqr.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


if __name__ == "__main__":
    gbqr = GradientBoostingQuantileRegressor(quantiles=[0.05, 0.5, 0.95])
    gbqr = gbqr.fit(X, y)
    y_hat = gbqr.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


if __name__ == "__main__":
    gbqr_casc = GradientBoostingQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(n_estimators=102),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=True,
    )
    gbqr_casc = gbqr_casc.fit(X, y)
    y_hat = gbqr_casc.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


# ### LightGBM

if __name__ == "__main__":
    lgbmqr = LightGBMQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(n_estimators=101, seed=np.random.randint(100)),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=False,
    )
    lgbmqr = lgbmqr.fit(X, y)
    y_hat = lgbmqr.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


if __name__ == "__main__":
    lgbmqr_casc = LightGBMQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(n_estimators=101, seed=np.random.randint(100)),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=True,
    )
    lgbmqr_casc = lgbmqr_casc.fit(X, y)
    y_hat = lgbmqr_casc.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


# ### CatBoost

if __name__ == "__main__":
    cbqr = CatBoostQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(
            n_estimators=101, silent=True, random_seed=np.random.randint(100)
        ),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=False,
    )
    cbqr = cbqr.fit(X, y)
    y_hat = cbqr.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))


if __name__ == "__main__":
    cbqr_casc = CatBoostQuantileRegressor(
        quantiles=[0.05, 0.5, 0.95],
        base_params=dict(
            n_estimators=101, silent=True, random_seed=np.random.randint(100)
        ),
        quantile_params={0.5: dict(n_estimators=200)},
        cascade=True,
    )
    cbqr_casc = cbqr_casc.fit(X, y)
    y_hat = cbqr_casc.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))
