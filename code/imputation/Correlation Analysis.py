#!/usr/bin/env python
# coding: utf-8

# # Correlation Analysis
# This script provides visualizations of the relationship between important features/strata and the conditional distributions of reporting delay and dates of disease onset. Only for exploratory purposes.

# ## Imports

import pandas as pd
import numpy as np


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)


import plotly.io as pio

pio.renderers.default = "svg"


import plotly.graph_objects as go
import plotly.express as px


# ## Data Loading and Preparation

delay = pd.read_pickle("../../data/processed/delay_2020-05-06.pl")


# #### Summarize all observations which do not have known or binary gender

delay.loc[
    (delay["gender"] != "male") & (delay["gender"] != "female"), "gender"
] = "other"


# #### Remove observations which are too recent

delay["reporting_delay_hd"].quantile([0.05, 0.95])


delay["date_report"].sort_values(ascending=False)


delay = delay[
    delay["date_report"] < (delay["date_report"].max() - pd.Timedelta(14, "days"))
]


# #### Remove observations with negative reporting delay

delay = delay[
    (delay["reporting_delay_hd"] >= 0) | (delay["reporting_delay_hd"].isnull())
]


# ## Correlation Analysis

plotlabels = {
    "reporting_delay_hd": "Reporting Delay [days]",
    "date_confirmation": "Date of Confirmation at RKI",
    "date_onset": "Date of Onset of Disease",
    "date_report": "Date of Report of Disease",
    "week_confirmation": "Calendar Week of Confirmation",
    "week_onset": "Calendar Week of Onset",
    "gender": "Gender",
    "age": "Age",
    "state": "State",
    "county": "County",
    "weekday_confirmation": "Weekday of Confirmation at RKI",
    "weekday_onset": "Weekday of Onset of Disease",
    "weekday_report": "Weekday of Report of Disease",
}


delay["reporting_delay_hd"].describe()


px.histogram(delay, x="reporting_delay_hd").show(renderer="svg", width=900)


# ### Time

delay_agg = (
    delay.query("reporting_delay_hd>=0 & date_report>'2020-03-01'")
    .groupby(["date_report"])["reporting_delay_hd"]
    .describe(percentiles=[0.25, 0.75])
)


delay["date_report"].max()


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=delay_agg.index,
        y=delay_agg["25%"],
        mode="lines",
        fill=None,
        line_color="rgba(0, 0, 0, 0)",
        showlegend=True,
        name="25% percentile",
    )
)

fig.add_trace(
    go.Scatter(
        x=delay_agg.index,
        y=delay_agg["50%"],
        mode="lines",
        line_color="red",
        name="median",
        fill="tonexty",
        fillcolor="rgba(255, 0, 0, 0.1)",
    )
)


fig.add_trace(
    go.Scatter(
        x=delay_agg.index,
        y=delay_agg["75%"],
        fill="tonexty",  # fill area between trace0 and trace1
        mode="lines",
        line_color="rgba(0, 0, 0, 0)",
        fillcolor="rgba(255, 0, 0, 0.1)",
        showlegend=True,
        name="75% percentile",
    )
)


fig.update_layout(
    title="Distribution of Reporting Delay by Date of Report",
    xaxis=dict(title="Date of Report at Health Department"),
    yaxis=dict(title="Reporting Delay [days]"),
)

fig.show(renderer="svg", width=900)


# px.violin(delay,x="gender", y="reporting_delay_hd", title= "Distribution of Reporting Delay (by Gender)",
#       labels=plotlabels)

px.histogram(
    delay,
    x="reporting_delay_hd",
    facet_col="gender",
    title="Distribution of Reporting Delay (by Gender)",
    labels=plotlabels,
    histnorm="probability density",
).show(renderer="svg", width=900)


# px.violin(delay,x="age_group1", y="reporting_delay_hd", title= "Distribution of Reporting Delay (by Age)",
#       labels=plotlabels)

px.histogram(
    delay,
    x="reporting_delay_hd",
    facet_col="age_group1",
    facet_col_wrap=3,
    title="Distribution of Reporting Delay (by Age)",
    labels=plotlabels,
    histnorm="probability",
).show(renderer="svg", width=900, height=1000)


px.violin(
    delay,
    x="week_report",
    y="reporting_delay_hd",
    title="Distribution of Reporting Delay (by Week)",
    labels=plotlabels,
).show(renderer="svg", width=900)

# px.histogram(delay, x="reporting_delay_hd", facet_col="week_report", facet_col_wrap=3, title= "Distribution of Reporting Delay (by Week)",
#       labels=plotlabels, histnorm='probability').show(height=1000)


# px.violin(delay,x="weekday_report", y="reporting_delay_hd", title= "Distribution of Reporting Delay (by Weekday)",
#       labels=plotlabels)

px.histogram(
    delay,
    x="reporting_delay_hd",
    facet_col="weekday_report",
    facet_col_wrap=3,
    title="Distribution of Reporting Delay (by Weekday)",
    histnorm="probability",
).show(renderer="svg", width=900, height=1000)


delay.groupby(["age_group1", "gender"])["reporting_delay_hd"].mean().head(200)
