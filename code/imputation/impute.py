#!/usr/bin/env python
# coding: utf-8

# # Imputation Script for Production

# ## Imports

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Import own code from other directory
import sys

sys.path.append("../../code/imputation")

from imputation_methods import impute_pmm


# ## Logging

import logging

logger = logging.getLogger(name="IMPUTATION")
logging.basicConfig()
logger.setLevel(logging.INFO)


# ## Data Loading and Preparation

logger.info("Loading data.")


filename = "delay_2020-05-06"


delay = pd.read_pickle(f"../../data/processed/{filename}.pl")
if "id" in delay.columns:
    delay = delay.drop("id", axis=1)


# #### Summarize all observations which do not have known or binary gender

delay.loc[
    (delay["gender"] != "male") & (delay["gender"] != "female"), "gender"
] = "other"


# #### Remove observations with negative reporting delay

delay_neg = delay[delay["reporting_delay_hd"] < 0]
delay = delay[
    (delay["reporting_delay_hd"] >= 0) | (delay["reporting_delay_hd"].isnull())
]


# #### One-Hot Encoding (Dummy Variables)

logger.info("Encoding as One-Hot variables.")


from sklearn.preprocessing import OneHotEncoder


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
delay_dummy, enc, enc_cats = to_dummy(delay[delay_labels])


# ## Perform Imputation

logger.info("Performing imputation.")


# impute delay
delay_imputed = impute_pmm(
    delay_dummy,
    "reporting_delay_hd",
    regressor=RandomForestRegressor(n_estimators=10),
    k_pmm=5,
    n=3,
)
# round to next integer
delay_imputed = delay_imputed.round()
# compute day of onset by subtracting delay from day of report
onset_imputed = delay_imputed.apply(
    lambda x: delay.loc[delay_imputed.index, "day_report"] - x
)


# ## Factor imputed values back into original dataframe

logger.info("Reintegrating as delay dataframe.")


def as_ordered_weekday(col):
    return col.astype(pd.CategoricalDtype(ordered=True)).cat.reorder_categories(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )


# Concat columns
delay_imp = delay.copy()
for col in onset_imputed.columns:
    delay_imp[col] = np.nan
    delay_imp.loc[onset_imputed.index, col] = onset_imputed[col]

# Pivot to long format
delay_imp = delay_imp.melt(
    id_vars=delay_imp.drop(["day_onset"] + list(onset_imputed.columns), axis=1).columns,
    value_vars=["day_onset"] + list(onset_imputed.columns),
    var_name="imputation",
    value_name="day_onset",
)

# Drop empty day onset rows
delay_imp = delay_imp.dropna(subset=["day_onset"]).sort_values("day_onset")

# Compute derived values for imputed rows
delay_imp["imputation"] = delay_imp["imputation"].replace({"day_onset": "original"})
delay_imp["imputed"] = delay_imp["imputation"] != "original"
delay_imp["date_onset"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
    delay_imp["day_onset"], unit="days"
)
delay_imp["week_onset"] = delay_imp["date_onset"].dt.week
delay_imp["weekday_onset"] = as_ordered_weekday(delay_imp["date_onset"].dt.day_name())
delay_imp["reporting_delay_hd"] = delay_imp["day_report"] - delay_imp["day_onset"]
delay_imp["reporting_delay_rki"] = delay_imp["day_report_rki"] - delay_imp["day_onset"]


# ### Add observations with negative reporting delay

delay_final = pd.concat(
    [delay_imp, delay_neg.assign(imputation="original", imputed=False)], sort=False
)


# ## Export imputed dataset

# As CSV

logger.info("Exporting as CSV.")


for imp in delay_final["imputation"].unique():
    if imp != "original":
        delay_final.query(f"imputation=='original' | imputation=='{imp}'").to_csv(
            f"../../data/processed/{filename}_{imp}.csv", index=False
        )


# As pickle file

logger.info("Exporting as pickle.")


for imp in delay_final["imputation"].unique():
    if imp != "original":
        delay_final.query(f"imputation=='original' | imputation=='{imp}'").to_pickle(
            f"../../data/processed/{filename}_{imp}.pl"
        )
