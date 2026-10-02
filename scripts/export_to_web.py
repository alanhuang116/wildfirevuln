"""Export everything the website shows into web/bundle.json.

Every number on the page comes from this file, and every number in this file
comes from the fitted model or the validation record. Map cells are
aggregated to ~250 m and suppressed below 5 homes, so no individual structure
can be located from the published site.
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from wildfirevuln import product, taxonomy as T  # noqa: E402
from wildfirevuln.model import GH_W, GH_X, Design, VulnModel  # noqa: E402
from wildfirevuln.validate import draw_impute, gbm_fit_predict  # noqa: E402

DATA = os.path.join(ROOT, "data", "processed", "dins_structures.csv")
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(ROOT, "web", "bundle.json")
REPLAY = ["Eaton 2025", "Palisades 2025", "Camp 2018", "Caldor 2021",
          "CZU Lightning Cmplx 2020", "Dixie 2021", "Park 2024", "Glass 2020"]
KGRID = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]
CELL_M = 250.0


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def share_with_obs(model, df_rest, y_obs, u, var, z=1.2815516):
    """Fire-wide destroyed share: observed homes count as observed, the rest
    are predicted; the band comes from the event-effect posterior."""
    eta = model.linpred(df_rest)
    n = len(df_rest) + len(y_obs)
    sd = np.sqrt(var)
    mid = (y_obs.sum() + model.predict(df_rest, u_mean=u, u_var=var).sum()) / n
    lo = (y_obs.sum() + expit(eta + u - z * sd).sum()) / n
    hi = (y_obs.sum() + expit(eta + u + z * sd).sum()) / n
    return mid, lo, hi


def live_replay(d, sp, ev):
    tr, te = d[d.event != ev], d[(d.event == ev) & (d.asset == "residential")]
    trp, tep = product.apply(tr, sp), product.apply(te, sp)
    wf = VulnModel(product.design(sp)).fit(trp)
    seed = zlib.crc32(ev.encode())
    tr_d, te_d = draw_impute(tr, tr, seed), draw_impute(te, tr, seed + 1)
    p_gbm = gbm_fit_predict(product.apply(tr_d, sp), product.apply(te_d, sp), list(sp),
                            tr.burnt.to_numpy(float))
    order = np.argsort(te.entry_order.to_numpy(), kind="stable")
    y = te.burnt.to_numpy(float)
    lg = np.log(np.clip(p_gbm, 1e-6, 1 - 1e-6) / (1 - np.clip(p_gbm, 1e-6, 1 - 1e-6)))
    rows = []
    for k in KGRID:
        if k > len(te) - 50:
            break
        obs, rest = order[:k], order[k:]
        if k == 0:
            u, v = 0.0, wf.tau2
        else:
            u, v = wf.event_posterior(tep.iloc[obs])
        mid, lo, hi = share_with_obs(wf, tep.iloc[rest], y[obs], u, v)
        if k == 0:
            g = p_gbm.mean()
        else:
            target = (y[obs].sum() + 0.5) / (k + 1.0)
            a, b = -10.0, 10.0
            for _ in range(60):
                m = (a + b) / 2
                a, b = (m, b) if expit(lg[obs] + m).mean() < target else (a, m)
            g = (y[obs].sum() + expit(lg[rest] + (a + b) / 2).sum()) / len(te)
        rows.append([k, r(mid), r(lo), r(hi), r(g), r(y[obs].mean() if k else None)])
    return {"event": ev, "n": len(te), "truth": r(y.mean()), "tau": r(np.sqrt(wf.tau2)),
            "steps": rows}


def map_cells(d, ev):
    g = d[d.event == ev]
    lat0 = g.lat.mean()
    kx = 111320.0 * np.cos(np.radians(lat0))
    ix = np.floor(g.lon * kx / CELL_M).astype(int)
    iy = np.floor(g.lat * 110540.0 / CELL_M).astype(int)
    agg = g.assign(ix=ix, iy=iy).groupby(["ix", "iy"]).agg(
        n=("burnt", "size"), b=("burnt", "sum"))
    agg = agg[agg.n >= 5].reset_index()
    cells = [[r((row.ix + 0.5) * CELL_M / kx, 5), r((row.iy + 0.5) * CELL_M / 110540.0, 5),
              int(row.n), int(row.b)] for row in agg.itertuples()]
    return {"event": ev, "cells": cells, "cell_m": CELL_M,
            "suppressed_homes": int(len(g) - agg.n.sum()),
            "center": [r(g.lat.median(), 4), r(g.lon.median(), 4)]}


def main():
    d = pd.read_csv(DATA)
    sp = product.spec()
    dp = product.apply(d, sp)
    final = VulnModel(product.design(sp)).fit(dp)
    mj = final.to_json()

    # Spacing presets from the terciles of structures within 100 m, by asset.
    presets = {}
    for a, ga in d.groupby("asset"):
        q1, q2 = ga.n100.quantile(1 / 3), ga.n100.quantile(2 / 3)
        presets[a] = []
        for name, lo, hi in [("Scattered", 0, q1), ("Suburban", q1, q2), ("Dense", q2, 1e9)]:
            g = ga[(ga.n100 >= lo) & (ga.n100 < hi)]
            presets[a].append({"name": name, "n30": float(g.n30.median()),
                               "n100": float(g.n100.median()), "nn_m": r(g.nn_m.median(), 1),
                               "burnt": r(g.burnt.mean())})
    res = d[d.asset == "residential"]

    # Partial damage when not destroyed, by structure class.
    partial = {}
    nd = d[d.damage < 4]
    for s, g in nd.groupby("struct"):
        sh = g.damage.value_counts(normalize=True)
        partial[s] = {"edr": r(sum(sh.get(k, 0) * T.DAMAGE_RATIO[k] for k in range(4)), 5),
                      "bands": [r(sh.get(k, 0)) for k in range(4)]}

    # Fires for the "replay a past fire" control, with their fitted effects.
    ev_n = res.event.value_counts()
    replay_effects = [{"event": e, "u": r(final.events[e]), "n": int(ev_n[e]),
                       "rate": r(res[res.event == e].burnt.mean())} for e in REPLAY]

    # Parity cases for the JavaScript engine.
    rng = np.random.default_rng(7)
    com_idx = np.flatnonzero((dp.asset == "commercial").to_numpy())
    res_idx = np.flatnonzero((dp.asset == "residential").to_numpy())
    cases = dp.iloc[np.r_[rng.choice(res_idx, 28, replace=False),
                          rng.choice(com_idx, 12, replace=False)]]
    parity = [{"home": {c: row[c] for c in list(sp) + ["n30", "n100", "nn_m"]},
               "p_prior": r(float(final.predict(cases.iloc[[i]])[0]), 8),
               "eta": r(float(final.linpred(cases.iloc[[i]])[0]), 8)}
              for i, (_, row) in enumerate(cases.iterrows())]
    for p in parity:
        p["home"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in p["home"].items()}

    with open(os.path.join(ROOT, "data", "processed", "credits.json")) as fh:
        credits = json.load(fh)
    with open(os.path.join(REP, "bench_summary.json")) as fh:
        bench = json.load(fh)
    with open(os.path.join(ROOT, "data", "processed", "commercial.json")) as fh:
        commercial = json.load(fh)
    per_fire = pd.read_csv(os.path.join(REP, "bench_leave_fire_out.csv"))
    inter = pd.read_csv(os.path.join(REP, "bench_event_interval.csv"))
    live = pd.read_csv(os.path.join(REP, "bench_live_update.csv"))
    live_mean = live.groupby(["k", "model"])[["brier", "logloss", "auc", "share_err"]].mean()
    qa = pd.read_csv(os.path.join(REP, "qa_ingestion.csv"))

    # The survey paradox: what is recorded on destroyed vs surviving homes.
    paradox = {}
    for f in ["fence", "deck", "eaves", "siding"]:
        t = pd.crosstab(res[f], res.burnt, normalize="columns")
        paradox[f] = {lv: [r(t.loc[lv, 0]), r(t.loc[lv, 1])] for lv in t.index}

    print("live replays ...")
    replays = [live_replay(d, sp, ev) for ev in REPLAY]
    cells = [map_cells(d, ev) for ev in REPLAY]

    bundle = {
        "meta": {"built": pd.Timestamp.now().strftime("%Y-%m-%d"),
                 "n_homes": int(len(res)), "n_commercial": int((d.asset == "commercial").sum()),
                 "n_structures": int(len(d)), "n_fires": int(d.event.nunique()),
                 "years": [int(d.year.min()), int(d.year.max())],
                 "burnt_share": r(res.burnt.mean()),
                 "n_raw": int(qa.records.iloc[0])},
        "model": mj, "spec": sp, "gh": {"x": GH_X.tolist(), "w": GH_W.tolist()},
        "names": T.LEVEL_NAMES, "fields": {f: v[3] for f, v in T.FIELDS.items()},
        "presets": presets, "partial": partial, "replay_effects": replay_effects,
        "damage_ratio": T.DAMAGE_RATIO, "parity": parity,
        "credits": credits,
        "bench": {"summary": bench["summary"], "coverage80": bench["interval_coverage80"],
                  "n_folds": len(bench["folds"]),
                  "per_fire": per_fire[["event", "model", "n", "rate", "brier", "auc",
                                        "share_err"]].round(4).to_dict("records"),
                  "intervals": inter.round(4).to_dict("records"),
                  "live": [{"k": int(k), "model": m, **{c: r(v) for c, v in row.items()}}
                           for (k, m), row in live_mean.iterrows()]},
        "qa": qa.to_dict("records"), "paradox": paradox,
        "replays": replays, "cells": cells, "commercial": commercial,
        "cre_map": {k: list(v) for k, v in T.CRE_MAP.items()},
        "commercial_struct": T.COMMERCIAL_STRUCT,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
    print("presets", presets)
    print("partial", partial)
    print("tau", np.sqrt(final.tau2))
    for rp in replays:
        print(rp["event"], rp["truth"], rp["steps"][0], rp["steps"][min(5, len(rp["steps"]) - 1)])
    print("cells", [len(c["cells"]) for c in cells])


if __name__ == "__main__":
    main()
