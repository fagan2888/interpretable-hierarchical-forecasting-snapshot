#!/usr/bin/env python
# coding: utf-8

# # Imputation Methods
# This notebook contains custom implementations of imputation methods.
#
# **Currently implemented methods**:
# - Multiple Imputation via Stochastic Regression with Bayesian Ridge
# - Custom scikit-learn imputer for Stochastic Regression with Bayesian Ridge
# - Multiple Imputation via Predictive Mean Matching by Statsmodels (with OLS)
# - Multiple Imputation via Predictive Mean Matching (with arbitrary regressor class)
# - Custom scikit-learn imputer for Predictive Mean Matching (with arbitrary regressor class)

# ## Imports

import pandas as pd
import numpy as np


# ## Imputation of Disease Onset Date

from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, RegressorMixin

import statsmodels.api as sm


# ### Multiple Imputation using Bayesian Regression (DIY implementation)

# Perform multiple imputation by drawing from Gaussian


def samplePredictions_df(preds, stds, n=1, name="imputation"):
    """
    Sample from posterior normal distributions of predicted instances.
    
    Parameters
    ----------
    preds : array_like
        Predicted conditional means.
    stds : array_like
        Predicted conditional standard deviations.
    n : int
        Number of samples to draw.
    name : str
        Prefix for names of the sample columns.
        
    Returns
    -------
    out : DataFrame
        Sampled values with one column per sample, one row per instance.
        
    """
    samples = pd.DataFrame({"pred_mean": preds, "pred_std": stds}).apply(
        lambda x: np.random.normal(x.pred_mean, x.pred_std, size=n), axis=1
    )
    return pd.DataFrame(
        list(samples),
        index=stds.index,
        columns=[f"{name}_{(1+i):02d}" for i in np.arange(n)],
    )


def impute_regression(data, target, n=1):
    """
    Perform multiple imputation by drawing from posterior distribution of Bayesian ridge regression model.
    
    Parameters
    ----------
    data : DataFrame
        Data to use for imputation.
    target : str
        Column to impute.
    n : int
        Number of multiple imputations to perform.
        
    Returns
    -------
    out : DataFrame
        One column per imputation, indices of original rows.
        
    """
    # fit regression model without nas
    y = data.dropna()[target]
    X = data.dropna().drop(target, axis=1)
    regr = BayesianRidge()
    regr.fit(X, y)

    # predict measures
    preds, stds = regr.predict(
        data[data[target].isnull()].drop(target, axis=1), return_std=True
    )

    # sample from distribution
    return samplePredictions_df(preds, stds, n=n, name="imputation")


class RegressionImputer(BaseEstimator, RegressorMixin):
    """Custom scikit-learn estimator for imputation with Bayesian regression"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.regr_ = BayesianRidge()
        self.regr_.fit(X, y)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "regr_")
        except AttributeError:
            raise RuntimeError("Imputer must be fitted before prediction.")

        # predict measures
        preds, stds = self.regr_.predict(X, return_std=True)

        return (
            pd.DataFrame({"pred_mean": preds, "pred_std": stds})
            .apply(lambda x: np.random.normal(x.pred_mean, x.pred_std), axis=1)
            .round()
        )


# ### Predictive mean matching (DIY imputation)
# Perform multiple imputation by drawing randomly from the k closests instances with observed value, where closesness is defined, using a regression model, as the distance between the prediction for the missing value instance and the prediction for the observed value instance.


def draw_nearest(obs, pred_obs, pred_miss, k_pmm=5):
    """
    For each of the missing values, draw from the k nearest observations.
    
    This function is adapted from the statsmodels MICEData module.
    
    Parameters
    ----------
    obs : array_like
        Observed values of instances with observed values.
    pred_obs : array_like
        Predicted values of instances with observed values.
    pred_miss : array_like
        Predicted values of instances with missing values.
    k_pmm : int
        Number of nearest neighbours to randomly draw from.
        
    Returns
    -------
    out : array_like
        Imputed values of instances with missing values.
        
    """
    # Jointly sort the observed and predicted values for the
    # cases with observed values.
    ii = np.argsort(pred_obs)
    obs = obs[ii]
    pred_obs = pred_obs[ii]

    # Find the closest match to the predicted values for
    # cases with missing values.
    ix = np.searchsorted(pred_obs, pred_miss)

    # Get the indices for the closest k_pmm values on
    # either side of the closest index.
    ixm = ix[:, None] + np.arange(-k_pmm, k_pmm)[None, :]

    # Account for boundary effects
    msk = np.nonzero((ixm < 0) | (ixm > len(obs) - 1))
    ixm = np.clip(ixm, 0, len(obs) - 1)

    # Get the distances
    dx = pred_miss[:, None] - pred_obs[ixm]
    dx = np.abs(dx)
    dx[msk] = np.inf

    # Get the closest positions in ix, row-wise.
    dxi = np.argsort(dx, 1)[:, 0:k_pmm]

    def choose_imputed(obs, pred_miss, dxi, k_pmm):
        # Choose a column for each row.
        ir = np.random.randint(0, k_pmm, len(pred_miss))

        # Unwind the indices
        jj = np.arange(dxi.shape[0])
        ix = dxi[(jj, ir)]
        iz = ixm[(jj, ix)]

        return np.array(obs[iz]).squeeze()

    imputed_miss = choose_imputed(obs, pred_miss, dxi, k_pmm)

    return imputed_miss


def impute_pmm(data, target, regressor=BayesianRidge(), k_pmm=5, n=1):
    """
    Perform multiple imputation by predictive mean matching.
    
    Parameters
    ----------
    data : DataFrame
        Data to use for imputation.
    target : str
        Column to impute.
    regressor : sklearn Regressor
        Estimator object to use for regression.
    k_pmm : array_like
        Number of nearest neighbours to randomly draw from.
    n : int
        Number of multiple imputations to perform.
        
    Returns
    -------
    out : DataFrame
        One column per imputation, indices of original rows.
        
    """
    y = data.dropna()[target]
    X = data.dropna().drop(target, axis=1)

    obs = data[data[target].notnull()][target].to_numpy()

    # Fit regression model
    regressor.fit(X, y)

    # Predict values
    pred_obs = regressor.predict(data[data[target].notnull()].drop(target, axis=1))
    pred_miss = regressor.predict(data[data[target].isnull()].drop(target, axis=1))

    # Impute
    imputed_values = {
        f"imputation_{(1+i):02d}": draw_nearest(obs, pred_obs, pred_miss, k_pmm=k_pmm)
        for i in np.arange(n)
    }

    return pd.DataFrame(imputed_values, index=data[data[target].isnull()].index)


class PmmImputer(BaseEstimator, RegressorMixin):
    """
    Custom scikit-learn estimator for imputation with predictive mean matching.
    
    Parameters
    ----------
    regressor : sklearn Regressor
        Estimator object to use for regression.
    k_pmm : array_like
        Number of nearest neighbours to randomly draw from.
        
    """

    def __init__(self, regressor=BayesianRidge(), k_pmm=5):
        self.k_pmm = k_pmm
        self.regressor = regressor

    def fit(self, X, y=None):
        self.obs_ = y.to_numpy()

        # Fit regression model
        self.regressor.fit(X, y)

        # Predict observed values
        self.pred_obs_ = self.regressor.predict(X)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "regressor")
        except AttributeError:
            raise RuntimeError("Imputer must be fitted before prediction.")

        # Predict missing values
        pred_miss = self.regressor.predict(X)

        # Impute
        imputed_values = draw_nearest(self.obs_, self.pred_obs_, pred_miss)

        return imputed_values


# ### Multiple Imputation with Predictive Mean Matching (Statsmodel)


def impute_pmm_stats(target, data, k_pmm=5, n=1):
    data_anonymous = data.copy()
    data_anonymous.columns = [f"x{i}" for i in range(data_anonymous.shape[1])]
    imp = sm.MICEData(data_anonymous, k_pmm=k_pmm)
    return pd.DataFrame(
        {f"imputation_{i:02d}": imp.next_sample()["x0"].copy() for i in range(n)},
        index=data.index,
    )
