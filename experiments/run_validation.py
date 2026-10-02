"""Leave-one-fire-out benchmark.

Each fold withholds one fire entirely and trains on every other fire.
Folds: fires with at least 200 inspected homes and at least 20 destroyed and
20 not destroyed (so ranking metrics are defined).

Compared vulnerability assumptions
  constant      pooled destroyed share of the training fires
  struct_only   structure type only (how a construction/occupancy table works)
  checklist     equal-weight mitigation checklist: share of 8 surveyed features
                in the compliant state, through a fitted logistic link
  survey_glm    every DINS survey field, pooled logistic, no event structure
  gbm           gradient boosting on every survey field plus spacing
  wildfirevuln  graded features plus spacing, partially pooled fire effect,
                prediction integrated over the fire effect

Live-fire update: after the first k inspections of the withheld fire (in
data-entry order, which follows the inspection teams), WildfireVuln updates
the fire effect; GBM is given the same information by re-centring its
intercept on the observed share. Both are scored on the remaining homes.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from wildfirevuln import product  # noqa: E402
from wildfirevuln.validate import run_fold  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "dins_residential.csv")
REP = os.path.join(ROOT, "reports")


def main():
    d = pd.read_csv(DATA)
    sp = product.spec()
    print("product spec:", sp)
    g = d.groupby("event").burnt.agg(["size", "sum"])
    folds = g[(g["size"] >= 200) & (g["sum"] >= 20) & (g["size"] - g["sum"] >= 20)].index
    print("folds:", len(folds))
    res = Parallel(n_jobs=-1, verbose=0)(delayed(run_fold)(d, sp, ev) for ev in folds)
    rows = pd.DataFrame([r for x in res for r in x[0]])
    inter = pd.DataFrame([x[1] for x in res])
    live = pd.DataFrame([r for x in res for r in x[2]])
    calib = pd.concat([x[3] for x in res], ignore_index=True)

    rows.to_csv(os.path.join(REP, "bench_leave_fire_out.csv"), index=False)
    inter.to_csv(os.path.join(REP, "bench_event_interval.csv"), index=False)
    live.to_csv(os.path.join(REP, "bench_live_update.csv"), index=False)
    calib.to_csv(os.path.join(ROOT, "data", "processed", "oos_predictions.csv"), index=False)

    summ = rows.groupby("model")[["brier", "logloss", "auc", "share_err"]].mean()
    w = rows.assign(wb=rows.brier * rows.n).groupby("model")
    summ["brier_weighted"] = w.wb.sum() / w.n.sum()
    summ = summ.sort_values("brier")
    wins = rows.pivot(index="event", columns="model", values="brier")
    summ["fires_won_brier"] = [int((wins.idxmin(axis=1) == m).sum()) for m in summ.index]
    print(summ.round(4).to_string())
    cov = ((inter.rate >= inter.lo80) & (inter.rate <= inter.hi80)).mean()
    print(f"event share 80% interval coverage: {cov:.3f} over {len(inter)} fires")
    print(live.groupby(["k", "model"])[["brier", "logloss", "auc", "share_err"]]
          .mean().round(4).to_string())
    with open(os.path.join(REP, "bench_summary.json"), "w") as fh:
        json.dump({"summary": summ.reset_index().to_dict("records"),
                   "interval_coverage80": float(cov), "folds": list(folds),
                   "spec": sp}, fh, indent=1)


if __name__ == "__main__":
    main()
