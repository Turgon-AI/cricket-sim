"""`cricket model`: walk-forward ODI models -> data/models/*.parquet, contract-checked and fingerprinted.

Reads gold only. Writes:
  model_ball_win_probability   per ball: the batting side's win probability before and after it, and the
                               swing (win probability added), from a model trained on earlier seasons only
  model_ball_outcome           per ball: probabilities of dot, 1, 2, 3, 4, 6, wicket, extra, and expected runs
  model_team_elo               pre-match Elo of both sides of every ODI
  report_model_*               scores, calibration, checkpoints and folds (and one INTERNAL comparison with
                               Cricinfo's published win probability, never exported)
"""

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from .. import contracts
from ..build import _git_state, fingerprint
from ..paths import GOLD, MODELS, SILVER
from ..transaction import new_stage, publish, run_worker, writer_lock
from . import dataset, report, train

KEYS = ["match_id", "innings", "delivery_number"]


def _wp_after(df: pd.DataFrame) -> pd.Series:
    """The batting side's win probability after the ball: the next ball's 'before' within an innings; after the
    last ball of innings 1, the innings-1 model's WP for the state after that ball (wp_model_end), so the gap
    between the innings-1 and chase models at the break is credited to no ball; after the match's last ball,
    the result. NULL when the match had no result."""
    df = df.sort_values(KEYS)
    nxt_match = df["match_id"].shift(-1)
    nxt_inn = df["innings"].shift(-1)
    nxt_wp = df["wp_model"].shift(-1)
    same_match = nxt_match == df["match_id"]
    after = np.where(same_match & (nxt_inn == df["innings"]), nxt_wp,
                     np.where(same_match, df["wp_model_end"], df["batting_won"]))
    return pd.Series(after, index=df.index)


def _write(con, name: str, frame: pd.DataFrame, order: str) -> None:
    expected = contracts.get(name).column_names
    if set(frame.columns) != set(expected) or frame.columns.duplicated().any():
        raise ValueError(f"{name}: model output columns differ from contract")
    con.register("frame", pa.Table.from_pandas(frame[expected], preserve_index=False))
    con.execute(f"COPY (SELECT * FROM frame ORDER BY {order}) TO '{MODELS / name}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.unregister("frame")


def _cricinfo_comparison(con, clean: pd.DataFrame) -> pd.DataFrame:
    """INTERNAL: our scores next to Cricinfo's own pre-ball win probability on the same balls (it is the
    previous ball's published value, so it is pre-ball too). Never exported: Cricinfo-derived."""
    path = SILVER / "cricinfo_ball_pre.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["innings", "balls", "perspective", "model_brier", "cricinfo_brier"])
    ci = con.execute(f"SELECT match_id, innings, delivery_number, win_probability_before / 100 AS ci "
                     f"FROM read_parquet('{path}')").df()
    m = clean.merge(ci, on=KEYS)
    out = []
    for innings, g in m.groupby("innings"):
        g = g[g["ci"].notna()]
        if g.empty:
            continue
        # Cricinfo's perspective is undocumented: take whichever orientation tracks the batting side's result
        same = np.mean((g["ci"] - g["batting_won"]) ** 2) <= np.mean((1 - g["ci"] - g["batting_won"]) ** 2)
        p = g["ci"] if same else 1 - g["ci"]
        out.append((int(innings), len(g), "batting_side" if same else "bowling_side",
                    float(np.mean((g["wp_model"] - g["batting_won"]) ** 2)), float(np.mean((p - g["batting_won"]) ** 2))))
    return pd.DataFrame(out, columns=["innings", "balls", "perspective", "model_brier", "cricinfo_brier"]).round(5)


def run(first_season: int = train.FIRST_TEST_SEASON, min_train_matches: int = 300, allow_dirty: bool = False) -> None:
    with writer_lock(MODELS.parent):
        stage = new_stage(MODELS.parent, "model")
        for name in ("staging", "silver", "gold"):
            (stage/name).symlink_to(MODELS.parent/name, target_is_directory=True)
        run_worker(stage, "model", {"first_season": first_season, "min_train_matches": min_train_matches,
                                    "allow_dirty": allow_dirty})
        publish(MODELS.parent, stage, ["models"])


def _run(first_season: int, min_train_matches: int, allow_dirty: bool) -> None:
    code = _git_state()
    if code["dirty"] and not allow_dirty:
        raise SystemExit("working tree has uncommitted changes; commit first or pass --allow-dirty")
    build_run = json.loads((GOLD / "run.json").read_text())
    from ..integrity import verify_artifacts
    verify_artifacts(GOLD.parent, build_run, ("staging", "silver", "gold"))
    MODELS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    dataset.load(con)
    rows = con.execute("SELECT * FROM model_rows").df()
    print(f"modelling {rows['match_id'].nunique():,} ODIs, {len(rows):,} balls")
    pred, folds = train.walk_forward(rows, first_season, min_train_matches)
    if pred.empty:
        raise SystemExit("no season had enough earlier matches to train on")
    df = rows.merge(pred, on=KEYS)
    df["wp_after"] = _wp_after(df)
    r6 = lambda s: s.astype(float).round(6)  # noqa: E731  (fixed precision keeps outputs byte-stable)

    wp = df[KEYS + ["batting_team", "trained_through_season"]].copy()
    wp["evaluated"], wp["exclusion_reason"] = df["is_clean"], df["exclusion_reason"]
    # Rain-revised targets and shortened innings lack a reliable as-of target/horizon in Cricsheet.
    # Do not publish a numeric WP for those rows: the model's nominal state would be a guess.
    for col, values in (("wp_before", df["wp_model"]), ("wp_after", df["wp_after"]),
                        ("expected_remaining_runs", df["expected_remaining_runs"])):
        wp[col] = r6(values).where(df["is_clean"])
    wp["wpa"] = (wp["wp_after"] - wp["wp_before"]).where(df["is_clean"])
    _write(con, "model_ball_win_probability", wp, ", ".join(KEYS))

    oc = df[KEYS + ["trained_through_season"]].copy()
    for cls in dataset.CLASSES:
        oc[f"p_{cls}"] = r6(df[f"p_{cls}"]).where(df["is_clean"])
    oc["expected_runs"] = r6(df["expected_runs"]).where(df["is_clean"])
    oc["evaluated"], oc["exclusion_reason"] = df["is_clean"], df["exclusion_reason"]
    _write(con, "model_ball_outcome", oc, ", ".join(KEYS))
    _write(con, "model_team_elo", con.execute("SELECT * FROM elo").df().assign(elo_before=lambda d: d["elo_before"].round(3)),
           "match_id, team")

    clean = df[df["is_clean"]].reset_index(drop=True)
    _write(con, "report_model_scores", report.scores(clean), "target, slice, model, scope")
    _write(con, "report_model_calibration", report.calibration(clean), "target, slice, model, era, bin")
    _write(con, "report_model_checkpoints", report.checkpoints(clean), "checkpoint, model")
    fold_rows = []
    for f in folds:
        row = {k: v for k, v in f.items() if not k.startswith("coefficients")}
        for innings in (1, 2):
            for name, value in (f.get(f"coefficients_innings_{innings}") or {}).items():
                row[f"innings_{innings}_{name}"] = round(value, 6)
        fold_rows.append(row)
    folds_df = pd.DataFrame(fold_rows)
    for innings in (1, 2):  # same columns on every run, even when a fold had too little data to fit
        for name in train.WP_LINEAR:
            folds_df[f"innings_{innings}_{name}"] = folds_df.get(f"innings_{innings}_{name}", np.nan)
    _write(con, "report_model_folds", folds_df, "season")
    _write(con, "report_model_vs_cricinfo_internal", _cricinfo_comparison(con, clean), "innings")

    problems = contracts.lint()
    if problems:
        raise SystemExit("contracts do not lint:\n" + "\n".join(problems))
    checks = contracts.enforce(con, layers=("models",))
    tables = {f"models/{p.stem}": fingerprint(con, p) for p in sorted(MODELS.glob("*.parquet"))}
    run_info = {"build_fingerprint": build_run["build_fingerprint"], "code": code,
                "settings": {"first_season": first_season, "min_train_matches": min_train_matches,
                             "seed": train.SEED}, "contract_checks": checks, "tables": tables,
                "model_fingerprint": hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()}
    (MODELS / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"model fingerprint {run_info['model_fingerprint'][:16]}  ({len(tables)} tables, {checks['checks_passed']} checks)")
