#!/usr/bin/env python
"""
Script for evaluation of scan logs from literature research
-----------------------------------------------------------
This script processes literature research logs by creating a
CSV file and a readme file for overview over the literature
scan. Moreover, it extracts the bibtex info of all positively
scanned publications into a separate file.

"""
__author__ = "Adrian Lison"

import re

import pandas as pd


def scan_to_df(scan, finished=True):
    BODY_SEPARATOR = "#-------------------------\nDETAILS ------------------\n#-------------------------"
    BIBTEX_PATTERN = "(.+?) ?= ?\{(.*?)\}"
    PUBLICATION_PATTERN = "@[A-Za-z]+?\{.*?\}\n\n"

    scan_header = scan.split(sep=BODY_SEPARATOR)[0]
    scan_body = scan.split(sep=BODY_SEPARATOR)[1]

    header_infos = {k: v for k, v in re.findall(BIBTEX_PATTERN, scan_header)}
    header_infos["found"] = int(header_infos["found"])
    try:
        header_infos["kept_after_scan"] = int(header_infos["kept_after_scan"])
    except Exception:
        pass

    publications_raw = re.findall(PUBLICATION_PATTERN, scan_body, re.DOTALL)
    publications = [
        {k: v for k, v in re.findall(BIBTEX_PATTERN, publication)}
        for publication in publications_raw
    ]

    # check integrity
    if finished:
        # is number of found publications equal to reported number of publications?
        if len(publications) != header_infos["found"]:
            raise UserWarning(
                "The number of found publications does not match the reported number of publications."
            )

        # have all publications been tagged correctly with a "take" decision?
        for publication in publications:
            if "take" not in publication:
                raise UserWarning(
                    f"Publication {publication['title']} has not valid take tag"
                )

    # make dataframe
    publicationFrame = pd.DataFrame(publications)

    publicationFrame.columns = [col.lower() for col in publicationFrame.columns]

    if finished:
        take = pd.DataFrame(
            publicationFrame["take"]
            .apply(lambda x: re.findall("(.+?)\.(.*)", x)[0])
            .tolist()
        )
        publicationFrame["take"] = take[0]
        publicationFrame["take_explanation"] = take[1].str.strip()

        if set(publicationFrame["take"]) != set(["Yes", "No"]):
            raise AssertionError("Not all take tags could be classified as Yes or No.")

        publicationFrame["take"] = publicationFrame["take"] == "Yes"

        for col in publicationFrame.columns:
            publicationFrame.loc[publicationFrame[col].isnull(), col] = ""

    return publicationFrame, header_infos, publications_raw


def pub_to_md(pub):
    res = f"""### **{pub['title'].strip()}**\n{pub['author'].strip()}. ({pub['year']}). {pub['title'].strip()}. *{pub['journal'].strip()}*"""
    if pub["take_explanation"] != "":
        res += "\n\n**Note:** " + (pub["take_explanation"].strip())
    return res


def df_to_AccDis(pubframe):
    accepted = "\n\n".join(
        [pub_to_md(pub) for k, pub in pubframe[pubframe["take"]].iterrows()]
    )
    discarded = "\n\n".join(
        [pub_to_md(pub) for k, pub in pubframe[~pubframe["take"]].iterrows()]
    )
    return accepted, discarded


def read_file(path):
    with open(path, "r", encoding="latin-1") as f:
        file = f.read()
    return file


def write_file(text, path):
    with open(path, "w", encoding="latin-1") as f:
        f.write(text)


scan = read_file("20_04_16 Nowcasting/Scopus/scan.txt")

publicationFrame, header_infos, publications_raw = scan_to_df(scan)

publicationFrame.to_csv("20_04_16 Nowcasting/Scopus/bibtex_table.csv", index=False)

accepted, discarded = df_to_AccDis(publicationFrame)

# write README file
readme_text = f"""
# Literature Research Scan
**Source:** {header_infos['where']}\n
**Date:** {header_infos['date']}\n
**Search Terms:** {header_infos['terms']}\n
**Search Criteria:** {header_infos['criterion']}\n
**Results:** {header_infos['found']} publications were scanned, {header_infos['kept_after_scan']} were kept. {header_infos['notes']}
\n-----
## Accepted Publications
{accepted}

\n-----
## Discarded Publications
{discarded}
"""

write_file(readme_text, "20_04_16 Nowcasting/Scopus/scan.md")


import Levenshtein as lev


def already_tagged(new, already, threshold=4):
    new1 = new[["title"]].assign(key=1).reset_index()
    already1 = (
        already[["title", "take", "take_explanation"]].assign(key=1).reset_index()
    )
    compare = pd.merge(new1, already1, on="key", suffixes=("_new", "_already"))
    compare["titles"] = tuple(zip(compare["title_new"], compare["title_already"]))
    compare["dist"] = compare["titles"].apply(lambda x: lev.distance(x[0], x[1]))
    return compare[compare["dist"] <= threshold].reset_index()[
        [
            "index_new",
            "title_new",
            "title_already",
            "index_already",
            "dist",
            "take",
            "take_explanation",
        ]
    ]


scan1 = read_file("20_04_16 Nowcasting/Scopus/scan.txt")
scan2 = read_file("20_04_16 Nowcasting/EBSCO/scan.txt")
publicationFrame1, header_infos1, publications_raw1 = scan_to_df(scan1)
publicationFrame2, header_infos2, publications_raw2 = scan_to_df(scan2, finished=False)

already = already_tagged(publicationFrame2, publicationFrame1)


def update_scan(new_scan, new_raw, already, already_header_info):
    for _, v in already.iterrows():
        insert = (
            ("Yes." if v["take"] else "No.") + " " + v["take_explanation"]
        ).strip()
        first_seen = already_header_info["date"] + " " + already_header_info["where"]
        new_scan = new_scan.replace(
            new_raw[v["index_new"]],
            new_raw[v["index_new"]].replace(
                "\n}\n\n", f"\ntake={{{insert}}},\nfirst_seen={{{first_seen}}},\n}}\n\n"
            ),
        )
    return new_scan


scan2_new = update_scan(scan2, publications_raw2, already, header_infos1)

write_file(scan2_new, "20_04_16 Nowcasting/EBSCO/scan_new.txt")
