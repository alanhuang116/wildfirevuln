"""The shipped feature specification, derived from the credit grades.

A field enters the pricing model only if at least one of its levels is
graded 'supported'. Within a field, a level that is not supported is merged
into the reference level: it earns no credit and carries no surcharge.
Consumed attachments never enter. Structure type and year built come from
the inspection form and the parcel roll respectively and enter whenever any
of their levels is supported.
"""
from __future__ import annotations

import json
import os

from . import taxonomy as T
from .model import REF

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CREDITS = os.path.join(ROOT, "data", "processed", "credits.json")


def spec(path=CREDITS):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    keep = {}
    for r in rows:
        if r["grade"] == "supported":
            keep.setdefault(r["field"], []).append(r["value"])
    order = ["struct", "era"] + list(T.FIELDS)
    return {f: sorted(keep[f]) for f in order if f in keep}


def apply(df, sp):
    out = df.copy()
    for f, levels in sp.items():
        ok = set(levels) | {REF[f], T.UNKNOWN}
        out[f] = out[f].where(out[f].isin(ok), REF[f])
    return out
