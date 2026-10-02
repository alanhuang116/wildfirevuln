"""The shipped feature specification, derived from the credit grades.

A field enters the pricing model only if at least one of its levels is
graded 'supported'. Within a field, a level that is not supported is merged
into the reference level: it earns no credit and carries no surcharge.
Consumed attachments never enter. Structure type and year built come from
the inspection form and the parcel roll respectively and enter whenever any
of their levels is supported. Commercial occupancy classes always enter.
"""
from __future__ import annotations

import json
import os

from . import taxonomy as T
from .model import REF, Design

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CREDITS = os.path.join(ROOT, "data", "processed", "credits.json")


def spec(path=CREDITS):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    keep = {}
    for r in rows:
        if r["grade"] == "supported":
            keep.setdefault(r["field"], []).append(r["value"])
    # Commercial occupancy classes are rating factors, not credits: they
    # always enter, estimated jointly with the residential classes.
    keep.setdefault("struct", []).extend(T.COMMERCIAL_STRUCT)
    order = ["struct", "era"] + list(T.FIELDS)
    return {f: sorted(set(keep[f])) for f in order if f in keep}


def apply(df, sp):
    out = df.copy()
    for f, levels in sp.items():
        ok = set(levels) | {REF[f], T.UNKNOWN}
        out[f] = out[f].where(out[f].isin(ok), REF[f])
    return out


def design(sp):
    """The shipped design: graded fields, spacing, commercial interactions."""
    return Design(cats=list(sp), use_num=True, interact=True)
