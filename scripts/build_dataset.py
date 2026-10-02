"""Build the analysis table from the raw DINS pull.

Admissibility: fire hazard, residential or commercial/infrastructure category, damage assessed
(Inaccessible dropped), incident start 2018 or later (the detailed coding era;
earlier events mostly inspected damaged structures only), coordinates in
California. Neighbourhood features are computed against every inspected
structure of the same incident, outbuildings and commercial included.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from wildfirevuln import taxonomy as T  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "dins.csv")
OUT = os.path.join(ROOT, "data", "processed", "dins_structures.csv")
QA = os.path.join(ROOT, "reports", "qa_ingestion.csv")
MIN_YEAR = 2018


def local_xy(lat, lon):
    lat0 = np.nanmean(lat)
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540.0
    return np.c_[x, y]


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    qa = [("pulled from CAL FIRE DINS", len(raw))]
    raw = raw[raw.HAZARDTYPE.eq("Fire")]
    qa.append(("hazard type = fire", len(raw)))
    raw = raw[raw.LATITUDE.between(32, 42.5) & raw.LONGITUDE.between(-125, -114)]
    qa.append(("coordinates inside California", len(raw)))
    raw["year"] = pd.to_datetime(raw.INCIDENTSTARTDATE, errors="coerce").dt.year
    raw = raw[raw.year >= MIN_YEAR]
    qa.append((f"incident start {MIN_YEAR} or later", len(raw)))
    raw["event"] = raw.INCIDENTNAME.str.strip() + " " + raw.year.astype(int).astype(str)
    raw["burnt"] = raw.DAMAGE.eq("Destroyed (>50%)").astype(float)
    raw.loc[raw.DAMAGE.eq("Inaccessible"), "burnt"] = np.nan

    parts = []
    for ev, g in raw.groupby("event"):
        xy = local_xy(g.LATITUDE.to_numpy(), g.LONGITUDE.to_numpy())
        tree = cKDTree(xy)
        n30 = np.array([len(i) - 1 for i in tree.query_ball_point(xy, 30.0)])
        n100 = np.array([len(i) - 1 for i in tree.query_ball_point(xy, 100.0)])
        dist, _ = tree.query(xy, k=2)
        nn = np.minimum(dist[:, 1], 200.0)
        # Leave-self-out share of neighbours destroyed within 100 m. This is a
        # post-event exposure measure: used to estimate mitigation effects,
        # never as a predictor for a new event.
        burnt = g.burnt.fillna(0).to_numpy()
        ok = g.burnt.notna().to_numpy().astype(float)
        nb = tree.query_ball_point(xy, 100.0)
        kb = np.array([burnt[i].sum() for i in nb]) - burnt
        kn = np.array([ok[i].sum() for i in nb]) - ok
        ev_rate = np.nansum(burnt) / max(ok.sum(), 1)
        loc = (kb + 2 * ev_rate) / (kn + 2)
        parts.append(pd.DataFrame({"n30": n30, "n100": n100, "nn_m": nn,
                                   "local_burnt": loc, "local_n": kn},
                                  index=g.index))
    raw = raw.join(pd.concat(parts))

    d = raw[raw.STRUCTURECATEGORY.isin(T.RESIDENTIAL | T.COMMERCIAL)].copy()
    qa.append(("residential, commercial or infrastructure", len(d)))
    d = d[d.DAMAGE.isin(T.DAMAGE)]
    qa.append(("damage assessed (Inaccessible dropped)", len(d)))

    out = pd.DataFrame(index=d.index)
    out["event"] = d.event
    out["year"] = d.year.astype(int)
    out["county"] = d.COUNTY
    out["lat"] = d.LATITUDE
    out["lon"] = d.LONGITUDE
    out["damage"] = d.DAMAGE.map(T.DAMAGE).astype(int)
    out["burnt"] = (out.damage == 4).astype(int)
    out["asset"] = np.where(d.STRUCTURECATEGORY.isin(T.RESIDENTIAL), "residential", "commercial")
    st = d.STRUCTURETYPE.map(T.STRUCT)
    st = st.where(st.notna(), np.where(out["asset"] == "residential", "sfr_1", "com_1"))
    # A commercial-category record with a residential type code (or the
    # reverse) is a data-entry inconsistency; the category wins.
    res_codes = {"sfr_1", "sfr_2", "mobile", "motorhome", "multi"}
    st = np.where((out["asset"] == "commercial") & st.isin(res_codes) &
                  ~d.STRUCTURECATEGORY.eq("Mixed Commercial/Residential"), "com_1", st)
    st = np.where(d.STRUCTURECATEGORY.eq("Mixed Commercial/Residential"), "mixed", st)
    st = np.where(d.STRUCTURECATEGORY.eq("Infrastructure"), "infrastructure", st)
    out["struct"] = st
    for f, (col, mp, _, _) in T.FIELDS.items():
        out[f] = d[col].map(lambda v, mp=mp: T.harmonise(v, mp))
    out["era"] = d.YEARBUILT.map(T.era)
    out["year_built"] = d.YEARBUILT.where(d.YEARBUILT > 1800)
    for c in ["n30", "n100", "nn_m", "local_burnt", "local_n"]:
        out[c] = d[c]
    out["cause"] = d.WHATDIDFIRESTARTFROM.fillna("")
    out["entry_order"] = d.OBJECTID
    out.to_csv(OUT, index=False)
    qa.append(("analysis table", len(out)))
    pd.DataFrame(qa, columns=["step", "records"]).to_csv(QA, index=False)
    for s, n in qa:
        print(f"{n:>9,}  {s}")
    print("events:", out.event.nunique(), " burnt share:", round(out.burnt.mean(), 3))


if __name__ == "__main__":
    main()
