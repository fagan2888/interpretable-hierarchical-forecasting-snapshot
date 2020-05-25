#!/usr/bin/env python
# coding: utf-8

# # MAR Analysis of Missing Value of Disease Onset Date

# ## Imports and Settings

import pandas as pd
import numpy as np


pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


import plotly.io as pio

pio.renderers.default = "svg"


from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_validate


# ## Data Loading and Preparation

delay = pd.read_pickle("../../data/processed/delay_2020-04-15.pl")


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


delay_labels = ["week_report", "weekday_report", "age", "gender", "state"]
delay[delay_labels].isnull().sum()


delay_dummy, enc, enc_cats = to_dummy(delay[delay_labels])
y = delay["reporting_delay_hd"].isnull()


# ## Predict Missingness
# The approach to invalidate the *Missing at Random (MAR) - Hypothesis* is to try to predict missingness using a regression model. If that is possible with decent performance, there must be a bias in the covariates.

models = {
    "dummy": DummyClassifier(strategy="stratified"),
    "knn": KNeighborsClassifier(3),
    "decision_tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced"),
    "random forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1
    ),
}
scorings = ["accuracy", "precision", "recall", "f1"]


# #### Test prediction of missingness using other features

cv_results = pd.concat(
    [
        pd.DataFrame(
            cross_validate(model, delay_dummy, y, scoring=scorings, cv=5)
        ).assign(model=modelname)
        for modelname, model in models.items()
    ]
)
cv_results.groupby("model").mean()


# #### Test prediction of missingness using randomly permuted features (zero information)

delay_dummy_shuffled = delay_dummy.copy()
for c in delay_dummy_shuffled.columns:
    delay_dummy_shuffled[c] = delay_dummy_shuffled.loc[
        np.random.permutation(delay_dummy_shuffled.index), c
    ].to_numpy()


cv_results_shuffled = pd.concat(
    [
        pd.DataFrame(
            cross_validate(model, delay_dummy_shuffled, y, scoring=scorings, cv=5)
        ).assign(model=modelname)
        for modelname, model in models.items()
    ]
)
cv_results_shuffled.groupby("model").mean()


# ### Preliminary Conclusion
# The extremely weak results of classification attempts for missing values (dummy models without variable usage perform on par, performance scores with randomly permuted features are not worse) of the reported onset date indicate that the other covariates are not clearly related to the missingness. This does not imply that (a fraction of) the data may be missing with other patterns. Nevertheless, it can be rather savely assumed that there is no strong relationship between the other variables and the missingness of the date of disease onset.

# ### Deprecated code

# #### Logistic Regression with Scikit-Learn

from sklearn.linear_model import LogisticRegression


cl_miss = LogisticRegression(random_state=0, solver="lbfgs", max_iter=400).fit(
    X_dummy, y
)


pd.Series(cl_miss.coef_[0], index=X_dummy.columns)


# #### Classification with Decision Tree

from sklearn import tree
from sklearn.model_selection import cross_validate


indx = X_dummy.sample(frac=1).index
scorings = ["accuracy", "precision", "recall", "f1"]
learner = tree.DecisionTreeClassifier(max_depth=20, class_weight="balanced")
pd.DataFrame(
    cross_validate(learner, delay_dummy.loc[indx,], y[indx], scoring=scorings, cv=5)
)


clf = tree.DecisionTreeClassifier(max_depth=20, class_weight="balanced")
clf = clf.fit(delay_dummy, y)


pd.Series(clf.feature_importances_, index=delay_dummy.columns)


from sklearn import metrics


print(f"Precision: {metrics.precision_score(y,clf.predict(delay_dummy)):5f}")
print(f"Recall: {metrics.recall_score(y,clf.predict(delay_dummy)):5f}")
print(f"F1: {metrics.f1_score(y,clf.predict(delay_dummy)):5f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
fig = tree.plot_tree(
    clf,
    max_depth=2,
    class_names=["False", "True"],
    filled=True,
    feature_names=delay_dummy.columns,
)
