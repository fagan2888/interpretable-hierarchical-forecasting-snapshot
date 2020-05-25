#!/usr/bin/env python
# coding: utf-8

#  # Preprocessing of Delay Data for Covid19

#  ## Imports

import pandas as pd
import numpy as np
import seaborn as sns

import logging


#  ## Data Loading

logger = logging.getLogger(name="PREPROCESSING")
logging.basicConfig()
logger.setLevel(logging.INFO)


suffix = "2020-05-06"

logger.info("Reading file.")
covid = pd.read_csv(
    f"../../data/raw/covid_{suffix}.csv",
    sep=",",
    encoding="ISO-8859-1",
    low_memory=False,
)


#  ## Preprocessing

# Manually fix some errors
logger.info("Manually fixing data input errors.")

covid.loc[covid.query("Id==23107834").index, "Meldedatum"] = "2020-04-07"


logger.info("Converting all date columns to datetime format.")
datecols = [
    "Erkrankungsbeginn",
    "Meldedatum",
    "DatumEingangRKI1",
    "DatumIstFall",
]  # select all columns featuring a date
for col in datecols:
    covid[col] = pd.to_datetime(covid[col])


logger.info("Performing subselection of columns.")
delay = covid[
    ["Erkrankungsbeginn", "Meldedatum", "DatumEingangRKI1", "DatumIstFall"]
    + [
        "MeldeLandkreis",
        "MeldeLandkreisBundesland",
        "AlterBerechnet",
        "Altersgruppe1",
        "Altersgruppe2",
        "Geschlecht",
    ]
].sort_values("DatumIstFall")


#  ## Checking Missing Value

#  ### Place

print("")
print("Null values for Landkreis: " + str(delay["MeldeLandkreis"].isnull().sum()))
print(
    "Null values for Bundesland: "
    + str(delay["MeldeLandkreisBundesland"].isnull().sum())
)


#  ### Age

print("Null values for AlterBerechnet: " + str(delay["AlterBerechnet"].isnull().sum()))
print("Null values for Altersgruppe1: " + str(delay["Altersgruppe1"].isnull().sum()))
print("Null values for Altersgruppe2: " + str(delay["Altersgruppe2"].isnull().sum()))
# --> The age columns are fine, no null values.


#  ### Gender

print("\nGender value counts:")
print(delay["Geschlecht"].value_counts(dropna=False))

print("")


logger.info("Renaming and selecting columns.")
mapping = {
    "Erkrankungsbeginn": "date_onset",
    "Meldedatum": "date_report",
    "DatumEingangRKI1": "date_report_rki",
    "DatumIstFall": "date_confirmation",
    "MeldeLandkreis": "county",
    "MeldeLandkreisBundesland": "state",
    "AlterBerechnet": "age",
    "Altersgruppe1": "age_group1",
    "Altersgruppe2": "age_group2",
    "Geschlecht": "gender",
}
delay = delay.rename(columns=mapping)[mapping.values()]


logger.info("Renaming column values.")
delay["gender"] = delay["gender"].replace(
    {
        "männlich": "male",
        "weiblich": "female",
        "divers": "diverse",
        "-nicht ermittelbar-": "undeterminable",
        "-nicht erhoben-": "unknown",
    }
)


logger.info("Adding derived values.")

datecols = [
    x for x in delay.columns if "date" in x.lower()
]  # select all columns featuring a date

# Add days since 2020-01-01 for all date columns
for col in datecols:
    delay[col.replace("date", "day")] = (
        delay[col] - pd.to_datetime("2020-01-01")
    ).dt.days

# Add calender week for all date columns
for col in datecols:
    delay[col.replace("date", "week")] = delay[col].dt.week

# Add day of the week for all date columns
def as_ordered_weekday(col):
    return col.astype(pd.CategoricalDtype(ordered=True)).cat.reorder_categories(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )


for col in datecols:
    delay[col.replace("date", "weekday")] = as_ordered_weekday(delay[col].dt.day_name())

# Add reporting delays
delay["reporting_delay_hd"] = (
    delay["day_report"] - delay["day_onset"]
)  # reporting delay health department
delay["reporting_delay_rki"] = (
    delay["day_confirmation"] - delay["day_onset"]
)  # reporting delay rki


logger.info("Exporting delay data to CSV.")
delay.to_csv(f"../../data/processed/delay_{suffix}.csv", index=False)

logger.info("Exporting delay data to pickle file.")
delay.to_pickle(f"../../data/processed/delay_{suffix}.pl")
