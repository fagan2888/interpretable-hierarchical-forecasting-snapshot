#!/usr/bin/env python
# coding: utf-8

# # Quantile Regression with Linear Models

import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin


import statsmodels.formula.api as smf
import statsmodels.api as sm


class LinearQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper for statsmodels Quantile Regression which provides functionality to jointly predict several quantiles.
    
    An independent model is used for each quantile.
    
    Parameters
    ----------
    quantiles : iterable
        Quantiles to predict, should be in ascending order.
    """

    def __init__(self, quantiles):
        self.quantiles = quantiles

    def fit(self, X, y):
        y = np.array(y)
        X = (
            sm.add_constant(np.array(X))
            if X.shape[0] > 1
            else np.hstack((np.ones((1, 1)), X))
        )
        mdl = sm.regression.quantile_regression.QuantReg(y, X)
        self.models_ = {q: mdl.fit(q) for q in self.quantiles}

        return self

    def predict(self, X):
        X = (
            sm.add_constant(np.array(X))
            if X.shape[0] > 1
            else np.hstack((np.ones((1, 1)), X))
        )
        predictions = np.array(
            [m.predict(X) for m in self.models_.values()]
        ).transpose()

        return predictions


# ## Tests

if __name__ == "__main__":
    import pandas as pd

    noro_ts = pd.read_pickle("../../data/processed/norovirus-ts.pl")


if __name__ == "__main__":
    input_window = 4
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


if __name__ == "__main__":
    lqr = LinearQuantileRegressor(quantiles=[0.05, 0.5, 0.95])
    lqr = lqr.fit(X, y)
    y_hat = lqr.predict(X)
    # Compute in-sample MAE for median
    print(np.mean(np.abs(y_hat[:, 1] - y)))
