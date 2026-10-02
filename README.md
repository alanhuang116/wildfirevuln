# WildfireVuln

Wildfire vulnerability for insurers and lenders: P(structure destroyed | fire
reaches its neighbourhood) for homes and commercial collateral, plus graded
mitigation credits, calibrated on the CAL FIRE Damage Inspection (DINS) record.

**Site:** https://alanhuang116.github.io/wildfirevuln/ (built into `docs/` by `scripts/build_site.py`)

## Data

| Step | Records |
|---|---|
| Pulled from the CAL FIRE DINS public feature service | 132,522 |
| Fire hazard, damage assessed, incident 2018 or later: residential | 77,682 homes |
| ... commercial, institutional, mixed use, infrastructure | 6,218 buildings |
| Fires | 351 |

Street addresses and parcel numbers are dropped at download. Structure-level
tables stay in `data/` and are not committed. The site shows aggregated
250 m cells only, and suppresses any cell with fewer than 5 homes.

## Findings

1. **Post-fire surveys record what is left.** Attached fences and decks that
   burned get recorded as "none", and eaves on burnt homes as "unknown" (52% of
   destroyed homes vs 26% of survivors). A model fitted to the survey naively
   says an attached wooden fence lowers the risk.
2. **"Unknown" leaks the outcome.** Gradient boosting that can detect unknown
   fields scores AUC 0.864 on withheld fires. Fill unknowns with random draws
   from the known values and it drops to 0.687. Restricted to graded features
   it scores 0.630; WildfireVuln scores 0.629.
3. **Credits are graded.** Each feature is estimated three ways (full survey,
   complete records, standing homes), controlling for fire and local exposure.
   8 of 21 levels are supported, including built 2008 or later, enclosed eaves,
   ≤1/8" vent mesh and wood roofs (as a surcharge). Fences, decks and patio
   covers cannot be identified from post-fire surveys.
4. **The fire matters more than the house.** The fire-to-fire spread is
   τ ≈ 1.5 on the logit scale. The 80% range for a new fire's destroyed share
   covers 26 of 31 withheld fires.
5. **Live updating.** After 25 inspections, Brier score on the rest of the
   fire falls from 0.231 to 0.197, beating re-centred boosting at every k tested.
6. **Commercial collateral.** DINS splits commercial buildings by storey count
   and a few institutional types; it has no warehouse, office or retail code, so
   those are scored as proxies. Feature effects differ for commercial buildings
   (post-2008 construction: -1.69 vs -0.62 log-odds), so the model gives them
   their own feature effects while sharing each fire's severity with homes. On
   16 withheld fires it ranks commercial buildings best (AUC 0.687), and after
   25 home inspections its commercial Brier is 0.191 against 0.208 for boosting.
   Before any inspections a commercial-only fit is better calibrated (0.212 vs
   0.220).

## Layout

```
wildfirevuln/   taxonomy, model (logistic + pooled fire effect), product spec, fold logic
scripts/        fetch, dataset build, web export, site build
experiments/    credits (graded, bootstrap), leave-one-fire-out validation, leakage ablation
reports/        benchmark CSVs and logs
web/            site template and data bundle
docs/           the published site
tests/          Python property tests, JS/Python engine parity
```

## Reproducing

```
python scripts/fetch_dins.py            # ~2 min, public ArcGIS service
python scripts/build_dataset.py
python experiments/run_credits.py       # bootstrap, ~5 min
python experiments/run_validation.py    # 31 folds, ~5 min
python experiments/run_commercial.py    # 16 folds + bootstrap, ~8 min
python scripts/export_to_web.py
python scripts/build_site.py
python tests/test_core.py
node tests/test_js_parity.js docs/index.html
```

Requires numpy, scipy, pandas, scikit-learn, joblib.

## Limits

California only; residential and commercial buildings, not sheds or barns.
Contents, stock, machinery and business interruption are not in DINS. The model conditions on fire reaching the
neighbourhood and needs a hazard model for how likely that is. Smoke, ash and
contents losses are not in DINS. Credits are associations within fires,
not randomised effects. No commercial catastrophe model was available to compare.

Independent analysis; not affiliated with or endorsed by CAL FIRE.
