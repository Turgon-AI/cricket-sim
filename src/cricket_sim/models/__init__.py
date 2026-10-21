"""Ball-level ODI models, trained and evaluated walk-forward (docs/VALIDATION.md, docs/MODEL_CARD.md).

  elo.py      pre-match team ratings, each computed only from earlier matches
  dataset.py  one row per ODI delivery: pre_ball inputs only, plus the targets
  train.py    walk-forward fits: win probability (per innings) and ball outcome (8 classes), with baselines
  report.py   proper scoring rules, calibration tables, checkpoint Brier scores
  run.py      `cricket model`: writes data/models/*.parquet, enforces their contracts, fingerprints them
"""
