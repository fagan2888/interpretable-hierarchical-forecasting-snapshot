#!/usr/bin/env python
# coding: utf-8

# # Validation of Missing Value Imputation Methods for Disease Onset Date
# Perform cross-validation of different methods for multiple imputation based on mean/median as well as on the posterior predictive distribution (using Kullback-Leibler divergence).

# ## Imports

import pandas as pd
import numpy as np

# KL-divergence measure
from scipy.stats import entropy


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)


# Plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "svg"


# Preprocessors
from sklearn.preprocessing import OneHotEncoder

# Imputers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Regressors
from sklearn.linear_model import BayesianRidge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

# Statsmodel API
import statsmodels.api as sm

# Cross-Validation
from sklearn.model_selection import cross_validate, cross_val_predict, RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Import own code from other directory
import sys

sys.path.append("../../code/imputation")

from imputation_methods import (
    impute_regression,
    RegressionImputer,
    impute_pmm,
    PmmImputer,
    impute_pmm_stats,
)


# ## Data Loading and Preparation

delay = pd.read_pickle("../../data/processed/delay.pl")


# #### Summarize all observations which do not have known or binary gender

delay.loc[
    (delay["gender"] != "male") & (delay["gender"] != "female"), "gender"
] = "other"


# #### Remove observations with negative reporting delay

delay = delay[
    (delay["reporting_delay_hd"] >= 0) | (delay["reporting_delay_hd"].isnull())
]


# #### One-Hot Encoding (Dummy Variables)


def to_dummy(X):
    enc = OneHotEncoder(handle_unknown="error", sparse=False, drop="first")
    X_cat = X.select_dtypes(include=[object, "category"])
    X_num = X.select_dtypes(exclude=[object, "category"])
    X_trans = pd.DataFrame(
        enc.fit_transform(X_cat),
        columns=enc.get_feature_names(X_cat.columns),
        index=X.index,
    )
    X_dummy = pd.concat([X_num, X_trans], axis=1)
    return X_dummy, enc, X_cat.columns


def from_dummy(X, enc, cat_columns):
    X_trans = X[enc.get_feature_names(cat_columns)]
    X_num = X.drop(enc.get_feature_names(cat_columns), axis=1)
    X_cat = pd.DataFrame(
        enc.inverse_transform(X_trans), columns=cat_columns, index=X.index
    )
    X_res = pd.concat([X_num, X_cat], axis=1)
    return X_res


def to_coded(X):
    X = X.copy()
    X_cat = X.select_dtypes(include=[object, "category"])

    # define
    def map_to_int(series):
        """Convert non-numeric features to integer codes"""
        series_cat = series.astype("category")
        mapping = dict(zip(series_cat, series_cat.cat.codes))
        return series_cat.cat.codes, mapping

    mappings = {feat: map_to_int(X[feat])[1] for feat in X_cat.columns}

    # apply
    for feat, mapping in mappings.items():
        X[feat] = X[feat].replace(mapping)

    return X, mappings


delay_labels = [
    "reporting_delay_hd",
    "week_report",
    "weekday_report",
    "age",
    "gender",
    "state",
]
delay[delay_labels].isnull().sum()


delay_dummy, enc, enc_cats = to_dummy(delay[delay_labels])


# ## Imputation of Disease Onset Date
# ### The following methods are tested:

# #### 1) Multiple Imputation using Bayesian Regression (DIY implementation)
# Perform multiple imputation by drawing from Gaussian

onset_imputed = impute_regression(delay_dummy, "reporting_delay_hd", n=3)


# #### 2) Predictive mean matching (DIY imputation) with different regression models
# Perform multiple imputation by drawing randomly from the k closests instances with observed value, where closesness is defined, using a regression model, as the distance between the prediction for the missing value instance and the prediction for the observed value instance.

onset_imputed = impute_pmm(delay_dummy, "reporting_delay_hd", n=3)


# # Cross-Validation of Imputation Approaches
# To compare the performance of different imputation models, cross-validation on observed data is performed.

regressors = {
    "Predictive Mean Matching BR5": (PmmImputer(k_pmm=5), True),
    "Predictive Mean Matching BR10": (PmmImputer(k_pmm=10), True),
    "Predictive Mean Matching RF2": (
        PmmImputer(regressor=RandomForestRegressor(n_estimators=10), k_pmm=2),
        True,
    ),
    "Predictive Mean Matching RF5": (
        PmmImputer(regressor=RandomForestRegressor(n_estimators=10), k_pmm=5),
        True,
    ),
    "Stochastic Regression": (RegressionImputer(), True),
    "Median": (DummyRegressor(strategy="median"), False),
    "Bayesian Ridge": (BayesianRidge(), True),
    "RF Regressor": (RandomForestRegressor(n_estimators=10), True),
}


def getXy(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


delay_cv = delay.loc[delay["reporting_delay_hd"] >= 0, delay_labels].dropna()
delay_dummy_cv, enc_dummy, enc_cats_dummy = to_dummy(delay_cv)
delay_codes_cv, enc_coded = to_coded(delay_cv)


# ## Cross-validation of imputed values based on mean / median
# The error metrics of the following cross-validation can only assess the accuracy of measures of location, not of the full distribution.

scorings = {
    "MAE": make_scorer(metrics.mean_absolute_error),
    "MRSE": make_scorer(
        lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    ),
    "r2": make_scorer(metrics.r2_score),
}

n_repeats = 3
n_splits = 5
cv_generator = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

log = True
df_models = []

for modelname, (model, use_dummy) in regressors.items():
    if log:
        print(f"Validating {modelname}")

    X, y = getXy(delay_dummy_cv if use_dummy else delay_codes_cv, "reporting_delay_hd")

    cv_results = cross_validate(
        model, X, y, scoring=scorings, cv=cv_generator, n_jobs=4
    )
    cv_results["model"] = modelname
    df_models.append(cv_results)

cv_results = pd.concat([pd.DataFrame(m) for m in df_models])
cv_results = cv_results[[list(cv_results.columns)[-1]] + list(cv_results.columns)[:-1]]
cv_results.groupby("model").mean()


# ## Cross-validation based on full posterior distribution

# #### Integration of CV-data from external approaches by Michael Hoehle

hoehle = pd.read_csv(
    "../../data/imputed_hoehle_3.csv",
    parse_dates=["rep_date", "disease_start", "disease_start_imp"],
    encoding="ISO-8859-1",
)
hoehle = hoehle.assign(
    reporting_delay_hd=lambda x: (x.rep_date - x.disease_start).dt.days,
    reporting_delay_hd_imp=lambda x: (x.rep_date - x.disease_start_imp).dt.days,
)  # estimated delay
# remove observations which are too recent
hoehle = hoehle.query("rep_date<'2020-04-14'")
# remove observations with negative reporting delay
hoehle = hoehle[
    (hoehle["reporting_delay_hd"] >= 0) | (hoehle["reporting_delay_hd"].isnull())
]

# compute estimated date of onset
hoehle = hoehle.assign(
    date_onset_imp=lambda x: (
        x.rep_date - pd.to_timedelta(x.reporting_delay_hd_imp, "days")
    )
)

# rename
hoehle = hoehle.rename(
    columns={
        "disease_start": "date_onset",
        "rep_date": "date_report",
        "sex": "gender",
        "Id": "id",
    }
)
hoehle["gender"] = hoehle["gender"].replace({"männlich": "male", "weiblich": "female"})
hoehle.loc[
    (hoehle["gender"] != "male") & (hoehle["gender"] != "female"), "gender"
] = "other"

datecols = [
    x for x in hoehle.columns if "date" in x.lower()
]  # select all columns featuring a date


# Add calender week for all date columns
for col in datecols:
    hoehle[col.replace("date", "week")] = hoehle[col].dt.week

# Add day of the week for all date columns
def as_ordered_weekday(col):
    return col.astype(pd.CategoricalDtype(ordered=True)).cat.reorder_categories(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )


for col in datecols:
    hoehle[col.replace("date", "weekday")] = as_ordered_weekday(
        hoehle[col].dt.day_name()
    )

hoehle["age_group1"] = (
    hoehle[["id", "age"]]
    .merge(delay[["age", "age_group1"]].drop_duplicates(), how="left")["age_group1"]
    .to_numpy()
)


hoehle_state = pd.read_csv(
    "../../data/imputed_hoehle_state.csv",
    parse_dates=["rep_date", "disease_start", "disease_start_imp"],
    encoding="ISO-8859-1",
)
hoehle_state = hoehle_state.assign(
    reporting_delay_hd=lambda x: (x.rep_date - x.disease_start).dt.days,
    reporting_delay_hd_imp=lambda x: (x.rep_date - x.disease_start_imp).dt.days,
)  # estimated delay
# remove observations which are too recent
hoehle_state = hoehle_state.query("rep_date<'2020-04-14'")
# remove observations with negative reporting delay
hoehle_state = hoehle_state[
    (hoehle_state["reporting_delay_hd"] >= 0)
    | (hoehle_state["reporting_delay_hd"].isnull())
]

# compute estimated date of onset
hoehle_state = hoehle_state.assign(
    date_onset_imp=lambda x: (
        x.rep_date - pd.to_timedelta(x.reporting_delay_hd_imp, "days")
    )
)

# rename
hoehle_state = hoehle_state.rename(
    columns={
        "disease_start": "date_onset",
        "rep_date": "date_report",
        "sex": "gender",
        "Id": "id",
    }
)
hoehle_state["gender"] = hoehle_state["gender"].replace(
    {"männlich": "male", "weiblich": "female"}
)
hoehle_state.loc[
    (hoehle_state["gender"] != "male") & (hoehle_state["gender"] != "female"), "gender"
] = "other"

datecols = [
    x for x in hoehle_state.columns if "date" in x.lower()
]  # select all columns featuring a date


# Add calender week for all date columns
for col in datecols:
    hoehle_state[col.replace("date", "week")] = hoehle_state[col].dt.week

# Add day of the week for all date columns
def as_ordered_weekday(col):
    return col.astype(pd.CategoricalDtype(ordered=True)).cat.reorder_categories(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )


for col in datecols:
    hoehle_state[col.replace("date", "weekday")] = as_ordered_weekday(
        hoehle_state[col].dt.day_name()
    )

hoehle_state["age_group1"] = (
    hoehle_state[["id", "age"]]
    .merge(delay[["age", "age_group1"]].drop_duplicates(), how="left")["age_group1"]
    .to_numpy()
)


# Define set of imputed values

use_all_folds = True
if use_all_folds:
    # using all folds
    (model, use_dummy) = regressors["Stochastic Regression"]
    X, y = getXy(delay_dummy_cv if use_dummy else delay_codes_cv, "reporting_delay_hd")
    imputed_reg = cross_val_predict(model, X, y, cv=3)

    (model, use_dummy) = regressors["Predictive Mean Matching BR5"]
    X, y = getXy(delay_dummy_cv if use_dummy else delay_codes_cv, "reporting_delay_hd")
    imputed_pmm_br = cross_val_predict(model, X, y, cv=3)

    (model, use_dummy) = regressors["Predictive Mean Matching RF5"]
    X, y = getXy(delay_dummy_cv if use_dummy else delay_codes_cv, "reporting_delay_hd")
    imputed_pmm_forest = cross_val_predict(model, X, y, cv=3)

    hoehle_imp = hoehle["reporting_delay_hd_imp"].to_numpy()
    hoehle_state_imp = hoehle_state["reporting_delay_hd_imp"].to_numpy()

    imputed = {
        "reg": imputed_reg,
        "pmm_br": imputed_pmm_br,
        "pmm_forest": imputed_pmm_forest,
    }
else:
    # using only one fold
    X, y = getXy(delay_dummy_cv, "reporting_delay_hd")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    (model, use_dummy) = regressors["Stochastic Regression"]
    model.fit(X_train, y_train)
    imputed_reg = model.predict(X_test)

    (model, use_dummy) = regressors["Predictive Mean Matching BR5"]
    model.fit(X_train, y_train)
    imputed_pmm_br = model.predict(X_test)

    (model, use_dummy) = regressors["Predictive Mean Matching RF5"]
    model.fit(X_train, y_train)
    imputed_pmm_forest = model.predict(X_test)

    hoehle_imp = hoehle.query("fold==1")["reporting_delay_hd_imp"].to_numpy()
    hoehle_state_imp = hoehle_state["reporting_delay_hd_imp"].to_numpy()

    imputed = {
        "reg": imputed_reg,
        "pmm_br": imputed_pmm_br,
        "pmm_forest": imputed_pmm_forest,
    }
    y = y_test


# ### Computation of Kullback-Leibler Divergence


def compute_kl(arr1, arr2, vrange):
    """
    Compute Kullback-Leibler divergence between two discrete distributions, given samples from each distribution.
    
    Parameters
    ----------
    arr1, arr2 : array_like
        Arrays of samples from two distributions to compare.
    vrange : iterable
        Iterable with valid values from the support of the probability distributions.
        
    """
    arr1_c = (
        pd.Series(arr1)
        .round()
        .value_counts()
        .sort_index()
        .reindex(vrange)
        .fillna(0.00001)
    )
    arr2_c = (
        pd.Series(arr2)
        .round()
        .value_counts()
        .sort_index()
        .reindex(vrange)
        .fillna(0.00001)
    )
    return entropy(arr1_c, arr2_c)


def kl_div_strata_dummy(X, y, imputed, strata, remaining):
    """Compute Kullback-Leibler divergence on stratified dataset with one-hot encoding"""
    kl_results = dict()
    for k, v in strata.iterrows():
        sub_idx = (X[list(v.index)] == v).all(axis=1).to_numpy()
        if v.any():
            strat_name = strata.columns[list(v.astype(bool))][0]
        else:
            strat_name = remaining

        strat_res = {
            k_imp: compute_kl(y[sub_idx], imp[sub_idx], range(25))
            for k_imp, imp in imputed.items()
        }

        kl_results[strat_name] = strat_res
    return kl_results


def kl_div_strata_cat(X, y, imputed, strata):
    """Compute Kullback-Leibler divergence on stratified dataset without one-hot encoding"""
    kl_results = dict()
    for k, v in strata.iterrows():
        sub_idx = (X[list(v.index)] == v).all(axis=1).to_numpy()
        strat_name = v.iloc[0]

        strat_res = {
            k_imp: compute_kl(y[sub_idx], imp[sub_idx], range(25))
            for k_imp, imp in imputed.items()
        }

        kl_results[strat_name] = strat_res
    return kl_results


# #### All

pd.DataFrame(
    {
        "All": {
            k_imp: compute_kl(y, imp, range(25))
            for k_imp, imp in {
                **imputed,
                "hoehle": hoehle_imp,
                "hoehle_state": hoehle_state_imp,
            }.items()
        }
    }
).assign(Rank=lambda x: x.rank())


# #### Gender

strata_subs = ["gender_male", "gender_other"]
remaining = "gender_female"
strata = X[strata_subs].drop_duplicates()
a = pd.DataFrame(kl_div_strata_dummy(X, y, imputed, strata, remaining))
a = a.rename(columns={col: col.replace("gender_", "") for col in a.columns})

strata = hoehle[["gender"]].drop_duplicates()
h = pd.DataFrame(
    kl_div_strata_cat(
        hoehle,
        hoehle["reporting_delay_hd"],
        {"hoehle": hoehle["reporting_delay_hd_imp"]},
        strata,
    )
)

strata = hoehle_state[["gender"]].drop_duplicates()
hs = pd.DataFrame(
    kl_div_strata_cat(
        hoehle_state,
        hoehle_state["reporting_delay_hd"],
        {"hoehle_state": hoehle_state["reporting_delay_hd_imp"]},
        strata,
    )
)

print(
    {
        k: v.index[v][0]
        for k, v in (pd.concat([a, h, hs], sort=True).rank() == 1).iteritems()
    }
)
pd.merge(
    pd.concat([a, h, hs], sort=True),
    pd.concat([a, h, hs], sort=True).rank(),
    left_index=True,
    right_index=True,
    suffixes=["", "_rank"],
    sort=True,
)


# #### Weekday

strata_subs = [
    "weekday_report_Monday",
    "weekday_report_Saturday",
    "weekday_report_Sunday",
    "weekday_report_Thursday",
    "weekday_report_Tuesday",
    "weekday_report_Wednesday",
]
remaining = "weekday_report_Friday"
strata = X[strata_subs].drop_duplicates()
a = pd.DataFrame(kl_div_strata_dummy(X, y, imputed, strata, remaining))
a = a.rename(columns={col: col.replace("weekday_report_", "") for col in a.columns})

strata = hoehle[["weekday_report"]].drop_duplicates()
h = pd.DataFrame(
    kl_div_strata_cat(
        hoehle,
        hoehle["reporting_delay_hd"],
        {"hoehle": hoehle["reporting_delay_hd_imp"]},
        strata,
    )
)

strata = hoehle_state[["weekday_report"]].drop_duplicates()
hs = pd.DataFrame(
    kl_div_strata_cat(
        hoehle_state,
        hoehle_state["reporting_delay_hd"],
        {"hoehle_state": hoehle_state["reporting_delay_hd_imp"]},
        strata,
    )
)

print(
    {
        k: v.index[v][0]
        for k, v in (pd.concat([a, h, hs], sort=True).rank() == 1).iteritems()
    }
)
pd.merge(
    pd.concat([a, h, hs], sort=True),
    pd.concat([a, h, hs], sort=True).rank(),
    left_index=True,
    right_index=True,
    suffixes=["", "_rank"],
    sort=True,
)


# #### State

strata_subs = [
    "state_Bayern",
    "state_Berlin",
    "state_Brandenburg",
    "state_Bremen",
    "state_Hamburg",
    "state_Hessen",
    "state_Mecklenburg-Vorpommern",
    "state_Niedersachsen",
    "state_Nordrhein-Westfalen",
    "state_Rheinland-Pfalz",
    "state_Saarland",
    "state_Sachsen",
    "state_Sachsen-Anhalt",
    "state_Schleswig-Holstein",
    "state_Thüringen",
]
remaining = "state_Baden-Württemberg"
strata = X[strata_subs].drop_duplicates()
a = pd.DataFrame(kl_div_strata_dummy(X, y, imputed, strata, remaining))
a = a.rename(columns={col: col.replace("state_", "") for col in a.columns})

strata = hoehle[["state"]].drop_duplicates()
h = pd.DataFrame(
    kl_div_strata_cat(
        hoehle,
        hoehle["reporting_delay_hd"],
        {"hoehle": hoehle["reporting_delay_hd_imp"]},
        strata,
    )
)

strata = hoehle_state[["state"]].drop_duplicates()
hs = pd.DataFrame(
    kl_div_strata_cat(
        hoehle_state,
        hoehle_state["reporting_delay_hd"],
        {"hoehle_state": hoehle_state["reporting_delay_hd_imp"]},
        strata,
    )
)

print(
    {
        k: v.index[v][0]
        for k, v in (pd.concat([a, h, hs], sort=True).rank() == 1).iteritems()
    }
)
pd.merge(
    pd.concat([a, h, hs], sort=True),
    pd.concat([a, h, hs], sort=True).rank(),
    left_index=True,
    right_index=True,
    suffixes=["", "_rank"],
    sort=True,
)


# #### Age

strata = (
    delay.loc[delay["reporting_delay_hd"] >= 0, ["age_group1"]]
    .dropna()
    .drop_duplicates()
)
a = pd.DataFrame(
    kl_div_strata_cat(
        delay.loc[delay["reporting_delay_hd"] >= 0, ["age_group1"]].dropna(),
        y,
        imputed,
        strata,
    )
)

strata = hoehle[["age_group1"]].drop_duplicates()
h = pd.DataFrame(
    kl_div_strata_cat(
        hoehle,
        hoehle["reporting_delay_hd"],
        {"hoehle": hoehle["reporting_delay_hd_imp"]},
        strata,
    )
)

strata = hoehle_state[["age_group1"]].drop_duplicates()
hs = pd.DataFrame(
    kl_div_strata_cat(
        hoehle_state,
        hoehle_state["reporting_delay_hd"],
        {"hoehle_state": hoehle_state["reporting_delay_hd_imp"]},
        strata,
    )
)

print(
    {
        k: v.index[v][0]
        for k, v in (pd.concat([a, h, hs], sort=True).rank() == 1).iteritems()
    }
)
pd.merge(
    pd.concat([a, h, hs], sort=True),
    pd.concat([a, h, hs], sort=True).rank(),
    left_index=True,
    right_index=True,
    suffixes=["", "_rank"],
    sort=True,
)


# #### Week

strata = X[["week_report"]].drop_duplicates()
a = pd.DataFrame(kl_div_strata_cat(X[["week_report"]], y, imputed, strata))

strata = hoehle[["week_report"]].drop_duplicates()
h = pd.DataFrame(
    kl_div_strata_cat(
        hoehle,
        hoehle["reporting_delay_hd"],
        {"hoehle": hoehle["reporting_delay_hd_imp"]},
        strata,
    )
)


strata = hoehle_state[["week_report"]].drop_duplicates()
hs = pd.DataFrame(
    kl_div_strata_cat(
        hoehle_state,
        hoehle_state["reporting_delay_hd"],
        {"hoehle_state": hoehle_state["reporting_delay_hd_imp"]},
        strata,
    )
)

print(
    {
        k: v.index[v][0]
        for k, v in (
            pd.concat([a, h, hs], sort=True).rank()
            == pd.concat([a, h, hs], sort=True).rank().min()
        ).iteritems()
    }
)
pd.merge(
    pd.concat([a, h, hs], sort=True),
    pd.concat([a, h, hs], sort=True).rank(),
    left_index=True,
    right_index=True,
    suffixes=["", "_rank"],
    sort=True,
)


# #### KL divergence generalization error using scikit-learn's CV-scheme

# Custom KL divergence scorer
class kl_scorer:
    """
    Custom scikit-learn scorer for Kullback-Leibler divergence.
    
    Parameters
    ----------
    strata : DataFrame
        Single-row dataframe with stratum feature values as columns.
    pred_range : iterable
        Iterable with valid values from the support of the probability distributions.
        
    """

    def __init__(self, strata=None, pred_range=range(100)):
        self.strata = strata
        self.remaining = remaining
        self.pred_range = pred_range

    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict(X)
        if self.strata is not None:
            X_nondummy = from_dummy(X, enc_dummy, enc_cats_dummy)
            strat_idx = (X_nondummy[list(self.strata.index)] == self.strata).all(axis=1)
        else:
            strat_idx = list(range(len(y_true)))
        return compute_kl(y_true[strat_idx], y_pred[strat_idx], self.pred_range)


regressors = {
    "Predictive Mean Matching BR5": (PmmImputer(), True),
    "Predictive Mean Matching BR10": (PmmImputer(k_pmm=10), True),
    "Predictive Mean Matching RF5": (
        PmmImputer(regressor=RandomForestRegressor(n_estimators=10), k_pmm=5),
        True,
    ),
    "Stochastic Regression": (RegressionImputer(), True),
}


strata = (
    delay.loc[delay["reporting_delay_hd"] >= 0, ["gender"]].dropna().drop_duplicates()
)


delay_cv = delay.loc[delay["reporting_delay_hd"] >= 0, delay_labels].dropna()
delay_dummy_cv, enc_dummy, enc_cats_dummy = to_dummy(delay_cv)
delay_codes_cv, enc_coded = to_coded(delay_cv)


def getXy(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


scorings = {
    "MAE": make_scorer(metrics.mean_absolute_error),
    "MRSE": make_scorer(
        lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    ),
    "r2": make_scorer(metrics.r2_score),
    "KL-Divergence": kl_scorer(),
}

scorings = {
    str(v.to_dict()): kl_scorer(strata=v, pred_range=range(25))
    for k, v in strata.iterrows()
}

log = True
df_models = []

for modelname, (model, use_dummy) in regressors.items():
    if log:
        print(f"Validating {modelname}")

    X, y = getXy(delay_dummy_cv if use_dummy else delay_codes_cv, "reporting_delay_hd")

    cv_results = cross_validate(
        model, X, y, scoring=scorings, cv=3, n_jobs=None, verbose=0
    )
    cv_results["model"] = modelname
    df_models.append(cv_results)

cv_results = pd.concat([pd.DataFrame(m) for m in df_models])
cv_results = cv_results[[list(cv_results.columns)[-1]] + list(cv_results.columns)[:-1]]
cv_results.groupby("model").mean()


# ## Visualization of the delay distribution

fig = go.Figure()
fig.add_trace(go.Histogram(x=y, name="Original"))
fig.add_trace(go.Histogram(x=imputed_reg, name="Regr."))
fig.add_trace(go.Histogram(x=imputed_pmm_br, name="PMM_baysreg"))
fig.add_trace(go.Histogram(x=imputed_pmm_forest, name="PMM_forest"))
fig.add_trace(go.Histogram(x=hoehle_imp, name="Hoehle"))
fig.add_trace(go.Histogram(x=hoehle_state_imp, name="Hoehle State"))

# Overlay both histograms
fig.update_layout(
    barmode="group",
    bargap=0.3,
    title="Delay distribution of original and imputed data",
    xaxis=dict(range=[-1, 25]),
)
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show(width=1000, renderer="svg")


# ### Visualization of the Disease Onset Time Series

imputers = {
    k: v
    for k, v in regressors.items()
    if k
    in [
        "Stochastic Regression",
        "Predictive Mean Matching BR5",
        "Predictive Mean Matching RF5",
    ]
}


def to_onset_df(prediction, id_):
    day_onset_temp = delay.loc[delay_cv.index]["day_report"] - prediction
    date_onset_temp = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        day_onset_temp, unit="days"
    )

    return pd.DataFrame(dict(trace=id_, date_onset=date_onset_temp))


X, y = getXy(delay_dummy_cv, "reporting_delay_hd")
onset = pd.concat(
    [
        to_onset_df(cross_val_predict(imputer[0], X, y, cv=3), id)
        for id, imputer in imputers.items()
    ]
    + [
        pd.DataFrame(dict(trace="hoehle", date_onset=hoehle["date_onset_imp"])),
        pd.DataFrame(
            dict(trace="hoehle_state", date_onset=hoehle_state["date_onset_imp"])
        ),
        pd.DataFrame(
            dict(trace="original", date_onset=delay.loc[delay_cv.index]["date_onset"])
        ),
    ]
)
# pd.DataFrame(dict(trace="report",date_onset=delay.loc[delay_cv.index]["date_report"]))


onset_count = (
    onset.groupby(["date_onset", "trace"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
onset_count = onset_count.sort_values(["trace", "date_onset"])
onset_count["trace"] = (
    onset_count["trace"]
    .str.replace("Predictive Mean Matching", "PMM")
    .str.replace("Stochastic Regression", "Baysreg")
)

fig = px.line(
    onset_count,
    x="date_onset",
    y="count",
    title="Case Counts by Onset Date",
    labels={"date_onset": "Date"},
    color="trace",
    line_shape="hv",
    color_discrete_sequence=px.colors.qualitative.G10,
)
fig.update_layout(
    xaxis_range=[pd.to_datetime("2020-02-19"), pd.to_datetime("2020-04-14")]
)
fig.for_each_trace(
    lambda trace: trace.update(line=dict(width=3))
    if trace.name == "trace=original"
    else trace.update(line=dict(dash="dot"))
)

fig.show(width=1000, renderer="svg")


# #### KL divergence for disease onset

onset = onset.assign(
    day_onset=lambda x: (x.date_onset - pd.to_datetime("2020-01-01")).dt.days
)
{
    trace: compute_kl(
        onset.query("trace=='original'")["day_onset"],
        onset.query(f"trace=='{trace}'")["day_onset"],
        range(100),
    )
    for trace in onset["trace"].unique()
}


# ## Deprecated Imputation Code

# ### Single Imputation with Regression (Scikit-learn iterative imputer)

imp = IterativeImputer(
    random_state=0, estimator=BayesianRidge(), sample_posterior=True, max_iter=1
)


imp.fit(delay_dummy)


onset_imputed = (
    pd.DataFrame({"imputation_01": imp.transform(delay_dummy)[:, 0]})
    .round()
    .astype(int)
)


# ### Multiple Imputation with Predictive Mean Matching (Statsmodel)


def impute_pmm_stats(target, data, k_pmm=5, n=1):
    data_anonymous = data.copy()
    data_anonymous.columns = [f"x{i}" for i in range(data_anonymous.shape[1])]
    imp = sm.MICEData(data_anonymous, k_pmm=k_pmm)
    return pd.DataFrame(
        {f"imputation_{i:02d}": imp.next_sample()["x0"].copy() for i in range(n)},
        index=data.index,
    )


onset_imputed = impute_pmm_stats("reporting_delay_hd", delay_dummy, n=3)


onset_imputed
