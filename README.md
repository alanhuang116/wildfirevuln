# WildfireVuln

Wildfire vulnerability and mitigation credits for property insurers:
P(home destroyed | fire reaches its neighbourhood), calibrated on the CAL FIRE
Damage Inspection (DINS) record.

**Site:** https://alanhuang116.github.io/wildfirevuln/ (built into `docs/` by `scripts/build_site.py`)

## Data

| Step | Records |
|---|---|
| Pulled from the CAL FIRE DINS public feature service | 132,522 |
| Fire hazard, residential, damage assessed, incident 2018 or later | 78,091 homes, 346 fires |

Street addresses and parcel numbers are dropped at download. Structure-level
tables stay in `data/` and are not committed. The site shows aggregated
250 m cells only, and suppresses any cell with fewer than 5 homes.

## Findings

1. **Post-fire surveys record what is left.** Attached fences and decks that
   burned get recorded as "none", and eaves on burnt homes as "unknown" (52% of
   destroyed homes vs 26% of survivors). A model fitted to the survey naively
   says an attached wooden fence lowers the risk.
2. **"Unknown" leaks the outcome.** Gradient boosting that can detect unknown
   fields scores AUC 0.865 on withheld fires. Fill unknowns with random draws
   from the known values and it drops to 0.688. Restricted to graded features
   it scores 0.631, the same as WildfireVuln.
3. **Credits are graded.** Each feature is estimated three ways (full survey,
   complete records, standing homes), controlling for fire and local exposure.
   8 of 21 levels are supported, including built 2008 or later, enclosed eaves,
   ≤1/8" vent mesh and wood roofs (as a surcharge). Fences, decks and patio
   covers cannot be identified from post-fire surveys.
4. **The fire matters more than the house.** The fire-to-fire spread is
   τ ≈ 1.5 on the logit scale. The 80% range for a new fire's destroyed share
   covers 26 of 31 withheld fires.
5. **Live updating.** After 25 inspections, Brier score on the rest of the
   fire falls from 0.231 to 0.198, beating re-centred boosting at every k tested.

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
python scripts/export_to_web.py
python scripts/build_site.py
python tests/test_core.py
node tests/test_js_parity.js docs/index.html
```

Requires numpy, scipy, pandas, scikit-learn, joblib.

## Limits

California residential only. The model conditions on fire reaching the
neighbourhood and needs a hazard model for how likely that is. Smoke, ash and
contents losses are not in DINS. Credits are associations within fires,
not randomised effects. No commercial catastrophe model was available to compare.

Independent analysis; not affiliated with or endorsed by CAL FIRE.
