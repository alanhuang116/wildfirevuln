"""Mitigation credits, with an evidence grade for every feature level.

Identification: compare structures in the same fire that faced the same
local fire (leave-self-out share of neighbours destroyed within 100 m) and
differ in the feature. Three designs:

  full      all records; 'unknown' gets its own indicator so its outcome
            dependence is absorbed rather than spread over known levels
  complete  records with every field known
  standing  structures not destroyed; outcome = any damage vs none. Every
            feature is visible on a standing house, so recording is reliable,
            but conditioning on survival biases protective features upward.

Grades
  supported  full and complete agree in sign, both clear zero, the standing
             design does not clearly contradict, and the sign agrees with the
             physical expectation from fire-lab testing (if one exists)
  contested  any of those fails; shown, never credited
  consumed   attachments that burn with the house (fence, deck, patio cover)
             are recorded as 'none' after a loss; DINS cannot identify them
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from wildfirevuln import taxonomy as T  # noqa: E402
from wildfirevuln.model import Design, VulnModel  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "dins_structures.csv")
OUT_CSV = os.path.join(ROOT, "reports", "credits.csv")
OUT_JSON = os.path.join(ROOT, "data", "processed", "credits.json")
B = int(os.environ.get("WFV_BOOT", 100))
CONSUMED = {"deck", "patio", "fence"}
# Expected sign of the coefficient relative to the reference level, from fire
# testing (IBHS, NIST) and building-code intent. None = no physical prior.
PRIOR = {
    "roof=tile": -1, "roof=metal": -1, "roof=wood": +1,
    "eaves=enclosed": -1, "eaves=none": -1,
    "vents=fine": -1, "vents=none": -1, "vents=open": +1,
    "siding=noncomb": -1, "windows=multi": -1,
    "era=1990_2007": -1, "era=post2008": -1,
    "deck=none": -1, "patio=noncomb": -1, "patio=none": -1,
    "fence=noncomb": -1, "fence=none": -1,
}


def load():
    d = pd.read_csv(DATA)
    d = d[d.asset == "residential"].copy()   # credits are a homeowner product
    lb = d.local_burnt.clip(0.01, 0.99)
    d["local_logit"] = np.log(lb / (1 - lb))
    return d


def fit(df, y):
    m = VulnModel(Design(use_num=False, extra=["local_logit"], unknown="indicator"),
                  ridge=0.1, em_iter=6).fit(df, y)
    cols = m.design.columns
    return (pd.Series(m.beta, index=cols),
            pd.Series(np.sqrt(np.diag(m.cov))[1:], index=cols), m)


def boot_one(d, seed):
    rng = np.random.default_rng(seed)
    evs = d.event.unique()
    pick = rng.choice(evs, size=len(evs), replace=True)
    parts = []
    for j, e in enumerate(pick):
        g = d[d.event == e].copy()
        g["event"] = f"{e}#{j}"
        parts.append(g)
    bd = pd.concat(parts, ignore_index=True)
    b, _, _ = fit(bd, bd.burnt)
    return b


def grade(level, full, lo, hi, cc, cc_se, st, st_se):
    field = level.split("=")[0]
    if field in CONSUMED:
        return "consumed"
    s_full = np.sign(full)
    clear_full = lo > 0 or hi < 0
    clear_cc = abs(cc / cc_se) > 2 if np.isfinite(cc) else False
    st_contra = np.sign(st) == -s_full and abs(st / st_se) > 2
    prior = PRIOR.get(level)
    ok = (clear_full and clear_cc and np.sign(cc) == s_full and not st_contra
          and (prior is None or prior == s_full))
    return "supported" if ok else "contested"


def credit(beta, p0):
    """Relative reduction in P(destroyed) at baseline probability p0."""
    o = p0 / (1 - p0) * np.exp(beta)
    return 1 - (o / (1 + o)) / p0


def main():
    d = load()
    full, full_se, m_full = fit(d, d.burnt)
    st = d[d.damage < 4]
    stand, stand_se, _ = fit(st, (st.damage >= 1).astype(int))
    fields = list(T.FIELDS)
    cc = d[(d[fields] != T.UNKNOWN).all(axis=1)]
    comp, comp_se, _ = fit(cc, cc.burnt)
    print(f"full {len(d):,}  standing {len(st):,}  complete {len(cc):,}")

    boots = Parallel(n_jobs=-1)(delayed(boot_one)(d, s) for s in range(B))
    bm = pd.concat(boots, axis=1).reindex(full.index)
    lo, hi = bm.quantile(0.05, axis=1), bm.quantile(0.95, axis=1)

    rows = []
    for lv in full.index:
        if lv.endswith("=" + T.UNKNOWN) or "=" not in lv:
            continue
        f, v = lv.split("=")
        g = grade(lv, full[lv], lo[lv], hi[lv], comp.get(lv, np.nan),
                  comp_se.get(lv, np.nan), stand.get(lv, 0.0), stand_se.get(lv, 1.0))
        rows.append({
            "level": lv, "field": f, "value": v,
            "field_name": T.FIELDS[f][3] if f in T.FIELDS else
            {"struct": "Structure type", "era": "Year built"}[f],
            "value_name": T.LEVEL_NAMES[f][v],
            "reference": T.LEVEL_NAMES[f][T.FIELDS[f][2] if f in T.FIELDS else
                                         {"struct": "sfr_1", "era": "pre1990"}[f]],
            "beta_full": full[lv], "lo90": lo[lv], "hi90": hi[lv],
            "odds_ratio": float(np.exp(full[lv])),
            "beta_complete": comp.get(lv, np.nan), "se_complete": comp_se.get(lv, np.nan),
            "beta_standing": stand.get(lv, np.nan), "se_standing": stand_se.get(lv, np.nan),
            "prior_sign": PRIOR.get(lv, 0), "grade": g,
            "credit_p30": credit(full[lv], 0.3),
            "credit_p30_lo": credit(hi[lv], 0.3), "credit_p30_hi": credit(lo[lv], 0.3),
            "n_level": int((d[f] == v).sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    exposure = {"beta": float(full["local_logit"]),
                "sd": float(m_full.design.sd["local_logit"])}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"rows": out.replace({np.nan: None}).to_dict("records"),
                   "exposure": exposure, "bootstrap": B,
                   "n": {"full": len(d), "standing": len(st), "complete": len(cc),
                         "events": int(d.event.nunique())}}, fh, indent=1)
    show = out[["level", "beta_full", "lo90", "hi90", "beta_complete",
                "beta_standing", "grade", "credit_p30"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
