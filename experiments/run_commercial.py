"""Commercial and infrastructure assets: does the joint model work for CRE?

1. Leave-one-fire-out on commercial buildings. Folds: fires with at least
   30 commercial structures, 5 destroyed and 5 not destroyed.
     constant          training commercial destroyed share
     home_curve        residential-only model applied as if every building
                       were a one-storey house (using home curves for CRE)
     commercial_only   the same model form fitted to commercial records only
     gbm_joint         boosting on graded features, all assets, leak-free
     wildfirevuln      joint model, all assets, shared fire effect
2. Transfer during an event: home inspections usually arrive first. After k
   home inspections, the shared fire effect is updated and the commercial
   buildings of that fire are scored.
3. Occupancy relativities with 90% fire-clustered bootstrap intervals.
4. Feature check: building-feature effects fitted on commercial records only,
   compared with the residential estimates.
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from wildfirevuln import product, taxonomy as T  # noqa: E402
from wildfirevuln.commercial import fold, boot_relativities  # noqa: E402
from wildfirevuln.model import Design, VulnModel  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "dins_structures.csv")
REP = os.path.join(ROOT, "reports")
B = int(os.environ.get("WFV_BOOT", 60))


def main():
    d = pd.read_csv(DATA)
    sp = product.spec()
    c = d[d.asset == "commercial"]
    g = c.groupby("event").burnt.agg(["size", "sum"])
    folds = list(g[(g["size"] >= 30) & (g["sum"] >= 5) & (g["size"] - g["sum"] >= 5)].index)
    print("commercial folds:", len(folds), " structures:", int(g.loc[folds, "size"].sum()))

    res = Parallel(n_jobs=-1)(delayed(fold)(d, sp, ev) for ev in folds)
    rows = pd.DataFrame([r for x in res for r in x[0]])
    inter = pd.DataFrame([x[1] for x in res])
    live = pd.DataFrame([r for x in res for r in x[2]])
    rows.to_csv(os.path.join(REP, "bench_commercial.csv"), index=False)
    inter.to_csv(os.path.join(REP, "bench_commercial_interval.csv"), index=False)
    live.to_csv(os.path.join(REP, "bench_commercial_live.csv"), index=False)
    summ = rows.groupby("model")[["brier", "logloss", "auc", "share_err"]].mean().sort_values("brier")
    print(summ.round(4).to_string())
    cov = float(((inter.rate >= inter.lo80) & (inter.rate <= inter.hi80)).mean())
    print(f"commercial share 80% coverage {cov:.3f} over {len(inter)} fires")
    lsum = live.groupby(["k", "model"])[["brier", "logloss", "share_err"]].mean()
    print(lsum.round(4).to_string())

    # Relativities: full joint model plus a fire-clustered bootstrap.
    dp = product.apply(d, sp)
    full = VulnModel(product.design(sp)).fit(dp)
    cols = full.design.columns
    est = pd.Series(full.beta, index=cols)
    boots = Parallel(n_jobs=-1)(delayed(boot_relativities)(dp, list(sp), s) for s in range(B))
    bm = pd.concat(boots, axis=1).reindex(cols)
    rel = []
    n_struct = d.struct.value_counts()
    for s in ["sfr_2", "mobile", "motorhome"] + T.COMMERCIAL_STRUCT:
        k = f"struct={s}"
        rel.append({"struct": s, "name": T.LEVEL_NAMES["struct"][s],
                    "asset": "commercial" if s in T.COMMERCIAL_STRUCT else "residential",
                    "odds_ratio": float(np.exp(est[k])),
                    "or_lo90": float(np.exp(bm.loc[k].quantile(0.05))),
                    "or_hi90": float(np.exp(bm.loc[k].quantile(0.95))),
                    "n": int(n_struct.get(s, 0)),
                    "burnt_share": float(d[d.struct == s].burnt.mean())})
    rel = pd.DataFrame(rel)
    rel.to_csv(os.path.join(REP, "commercial_relativities.csv"), index=False)
    print(rel.round(3).to_string(index=False))

    # Feature check: commercial-only fit vs residential-only fit, same form.
    feats = [f for f in sp if f != "struct"]
    out = []
    fits = {}
    for a in ["residential", "commercial"]:
        sub = product.apply(d[d.asset == a], sp)
        m = VulnModel(Design(cats=feats, use_num=True)).fit(sub)
        fits[a] = (pd.Series(m.beta, index=m.design.columns),
                   pd.Series(np.sqrt(np.diag(m.cov))[1:], index=m.design.columns))
    for k in fits["residential"][0].index:
        if "=" not in k:
            continue
        br, sr = fits["residential"][0][k], fits["residential"][1][k]
        bc, sc = fits["commercial"][0].get(k, np.nan), fits["commercial"][1].get(k, np.nan)
        z = (bc - br) / np.sqrt(sr ** 2 + sc ** 2)
        out.append({"level": k, "beta_residential": br, "se_residential": sr,
                    "beta_commercial": bc, "se_commercial": sc, "z_difference": z})
    fc = pd.DataFrame(out)
    fc.to_csv(os.path.join(REP, "commercial_feature_check.csv"), index=False)
    print(fc.round(3).to_string(index=False))

    with open(os.path.join(ROOT, "data", "processed", "commercial.json"), "w") as fh:
        json.dump({"summary": summ.reset_index().to_dict("records"),
                   "coverage80": cov, "folds": folds,
                   "n_structures": int(len(c)),
                   "intervals": inter.round(4).to_dict("records"),
                   "live": lsum.reset_index().round(5).to_dict("records"),
                   "relativities": rel.round(4).to_dict("records"),
                   "feature_check": fc.round(4).to_dict("records"),
                   "bootstrap": B}, fh, indent=1)


if __name__ == "__main__":
    main()
