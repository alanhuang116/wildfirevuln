"""Pull the CAL FIRE Damage Inspection (DINS) master table.

Source: CAL FIRE POSTFIRE_MASTER_DATA_SHARE feature service (public).
Every structure inspected inside or near a fire perimeter, damaged or not.
Street-level address fields are dropped at ingest; coordinates are kept for
neighbourhood features and never leave data/raw.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
       "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "raw", "dins.csv")

DROP = {"GLOBALID", "STREETNUMBER", "STREETNAME", "STREETTYPE", "STREETSUFFIX",
        "SITEADDRESS", "APN"}
PAGE = 2000


def get(params, tries=5):
    q = URL + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print("  retry", k + 1, e)
            time.sleep(3 * (k + 1))


def main():
    n = get({"where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    print("records on server:", n)
    rows, off = [], 0
    while off < n:
        d = get({"where": "1=1", "outFields": "*", "returnGeometry": "false",
                 "orderByFields": "OBJECTID", "resultOffset": off,
                 "resultRecordCount": PAGE, "f": "json"})
        feats = d.get("features", [])
        if not feats:
            break
        rows.extend({k: v for k, v in f["attributes"].items() if k not in DROP}
                    for f in feats)
        off += len(feats)
        print(f"  {off:,}/{n:,}")
    df = pd.DataFrame(rows)
    df["INCIDENTSTARTDATE"] = pd.to_datetime(df["INCIDENTSTARTDATE"], unit="ms",
                                             errors="coerce")
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df))


if __name__ == "__main__":
    main()
