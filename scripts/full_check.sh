#!/usr/bin/env bash
# Full-data check. CI builds only the 10-match sample, so faults that need the real data
# (retired-not-out wickets, awarded matches, associate-team T20Is, a model input that leaks the result)
# pass CI. This script runs the whole pipeline on a LOCAL FULL data folder and then checks those cases.
#
#   scripts/full_check.sh [DATA_DIR]
#
# DATA_DIR is a data root holding bronze/ (ingested and verified). Default: $CRICKET_SIM_FULL_DATA, else the
# repo's own data/. It takes about 15 minutes. It stops at the first failing step (exit code non-zero).
# The sample variables CRICKET_SIM_DATA and CRICKET_SIM_SNAPSHOTS are ignored: this is always the full data and
# the repo's own config/ snapshot lock. No network is used.
#
# Steps: verify, build, model, contracts, export, then the full-data-only checks below.
# A dirty git tree is allowed (--allow-dirty is added, and recorded in run.json): this is a check, not a release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ARG="${1:-${CRICKET_SIM_FULL_DATA:-$ROOT/data}}"
[ -d "$DATA_ARG/bronze" ] || { echo "full_check: $DATA_ARG has no bronze/ folder; give a full data root" >&2; exit 2; }
DATA="$(cd "$DATA_ARG" && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
unset CRICKET_SIM_SNAPSHOTS
export CRICKET_SIM_DATA="$DATA"

ALLOW_DIRTY=""
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then ALLOW_DIRTY="--allow-dirty"; fi
echo "full_check: data root $DATA ${ALLOW_DIRTY:+(dirty tree: $ALLOW_DIRTY)}"

step() {  # step <name> <command...>: run it, show the time, stop at the first failure
  local name="$1"; shift
  local start=$SECONDS
  echo; echo "== $name"
  if ! "$@"; then
    echo "full_check: FAILED at step '$name'" >&2
    exit 1
  fi
  echo "-- $name ok ($((SECONDS - start))s)"
}

step "verify"    uv run cricket verify
step "build"     uv run cricket build $ALLOW_DIRTY
step "model"     uv run cricket model $ALLOW_DIRTY
step "contracts" uv run cricket contracts

# An export folder is named from the data fingerprint. If one with that name exists and differs (for example
# it was made from another commit), `cricket export` refuses to replace it, so move it aside first (never delete).
move_old_export() {
  local name
  name="$(uv run python -I -c "import json; print(json.load(open('$DATA/gold/run.json'))['build_fingerprint'][:12])")"
  if [ -d "$DATA/export/$name" ]; then
    mkdir -p "$DATA/export-replaced"
    mv "$DATA/export/$name" "$DATA/export-replaced/$name.$(date +%Y%m%d-%H%M%S)"
    echo "moved the previous export of this data aside to $DATA/export-replaced/"
  fi
}
do_export() { move_old_export; uv run cricket export $ALLOW_DIRTY; }
step "export"    do_export

full_data_checks() {
  uv run python -I - "$DATA" <<'PY'
import json
import sys
from pathlib import Path

import duckdb
import yaml

data = Path(sys.argv[1])
root = Path.cwd()
scope = yaml.safe_load((root / "config" / "scope.yml").read_text())
con = duckdb.connect()
failures = []


def one(sql):
    return con.execute(sql.replace("{d}", str(data))).fetchone()[0]


def check(ok, message):
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


print("retired not out")
retired_not_out = one("SELECT count(*) FROM '{d}/silver/wickets.parquet' WHERE kind = 'retired not out'")
check(retired_not_out > 0, f"the data has 'retired not out' wickets ({retired_not_out}); with none, this check proves nothing")
bad = one("""SELECT count(*) FROM '{d}/silver/wickets.parquet'
             WHERE (kind IN ('retired hurt', 'retired not out')) <> is_retirement""")
check(bad == 0, f"every retired hurt / retired not out wicket is a retirement, and nothing else is ({bad} wrong)")
bad = one("""SELECT count(*) FROM '{d}/gold/int_ball_state_pre.parquet' s JOIN '{d}/gold/int_innings_state.parquet' i
             USING (match_id, innings) WHERE NOT i.is_super_over AND s.innings_wickets_before > 9""")
check(bad == 0, f"no normal innings is ever more than 9 wickets down before a ball ({bad} balls)")

print("awarded matches")
awarded = one("SELECT count(*) FROM '{d}/silver/matches.parquet' WHERE method = 'Awarded'")
check(awarded > 0, f"the data has awarded matches ({awarded})")
bad = one("SELECT count(*) FROM '{d}/gold/int_match_quality.parquet' WHERE is_awarded AND passes_quality")
check(bad == 0, f"no awarded match passes quality ({bad} do)")
bad = one("""SELECT count(*) FROM '{d}/gold/int_match_quality.parquet' q JOIN '{d}/silver/matches.parquet' m USING (match_id)
             WHERE (m.method = 'Awarded') <> q.is_awarded""")
check(bad == 0, f"is_awarded is exactly 'Cricsheet method = Awarded' ({bad} differ)")

print("scope, recomputed from silver and config/scope.yml")
members = ", ".join("'" + t.replace("'", "''") + "'" for t in scope["full_members"])
expected = {}
for seg in scope["segments"]:
    conds = [f"gender = '{scope['gender']}'", f"start_date >= DATE '{scope['from_date']}'"]
    for key, value in seg.items():
        if key == "name":
            continue
        if key == "both_teams_in":
            conds.append(f"team1 IN ({members}) AND team2 IN ({members})")
        else:
            conds.append(f"{key} = '{value}'")
    for (match_id,) in con.execute(f"SELECT match_id FROM '{data}/silver/matches.parquet' WHERE " + " AND ".join(conds)).fetchall():
        expected.setdefault(match_id, seg["name"])
actual = dict(con.execute(f"SELECT match_id, segment FROM '{data}/gold/dim_match.parquet'").fetchall())
check(expected == actual, f"dim_match holds exactly the matches the scope selects ({len(actual)}; "
      f"{len(set(expected) - set(actual))} missing, {len(set(actual) - set(expected))} extra)")
t20i_bad = one(f"""SELECT count(*) FROM '{data}/gold/dim_match.parquet'
                   WHERE segment = 'T20I' AND NOT (team1 IN ({members}) AND team2 IN ({members}))""")
check(t20i_bad == 0, f"every T20I has two full-member teams ({t20i_bad} do not)")
one_member = {r[0] for r in con.execute(f"""SELECT match_id FROM '{data}/silver/matches.parquet'
        WHERE gender = '{scope['gender']}' AND match_type = 'T20' AND team_type = 'international'
          AND (team1 IN ({members})) <> (team2 IN ({members}))""").fetchall()}
check(one_member and not one_member & set(actual),
      f"the {len(one_member)} T20Is with only one full-member side are all left out of dim_match")

print("headline and reports against the facts")
headline = json.loads((data / "gold" / "headline.json").read_text())
matches = one("SELECT count(*) FROM '{d}/gold/dim_match.parquet'")
balls = one("SELECT count(*) FROM '{d}/gold/fact_delivery.parquet'")
check(headline["matches_in_scope"] == matches, f"headline matches_in_scope {headline['matches_in_scope']} = dim_match rows {matches}")
check(headline["balls_in_scope"] == balls, f"headline balls_in_scope {headline['balls_in_scope']} = fact_delivery rows {balls}")
passing = one("SELECT count(*) FROM '{d}/gold/dim_match.parquet' WHERE passes_quality")
check(headline["matches_passing_quality_pct"] == round(100.0 * passing / matches, 2), "headline passing percentage is passing / matches")
check(99.0 <= headline["matches_passing_quality_pct"] <= 100.0,
      f"matches passing quality is in the expected band 99 to 100 ({headline['matches_passing_quality_pct']}%)")
by_year = one("SELECT sum(matches) FROM '{d}/gold/report_coverage_by_segment_year.parquet'")
check(by_year == matches, f"coverage report 'matches' adds up to the matches in scope ({by_year} = {matches})")
ipl = one("SELECT sum(matches) FROM '{d}/gold/report_ipl_seasons.parquet'")
check(ipl == one("SELECT count(*) FROM '{d}/gold/dim_match.parquet' WHERE segment = 'IPL'"), "IPL seasons report counts IPL matches only")

print("model scores: a leaked input would make them far too good")
scores = {(t, s, m): (ll, br) for t, s, m, ll, br in con.execute(
    f"SELECT target, slice, model, log_loss, brier FROM '{data}/models/report_model_scores.parquet' WHERE scope = 'all'").fetchall()}
for target, slice_ in (("ball_outcome", "all_classes"), ("win_probability", "innings_1"), ("win_probability", "innings_2")):
    ll, br = scores[(target, slice_, "model")]
    base_ll, base_br = scores[(target, slice_, "base_rate")]
    check(0.3 * base_br < br < base_br, f"{target} {slice_}: Brier {br} is better than the base rate {base_br} but not by a leak's margin")
    check(0.3 * base_ll < ll < base_ll, f"{target} {slice_}: log loss {ll} is better than the base rate {base_ll} but not by a leak's margin")

if failures:
    print(f"\n{len(failures)} full-data check(s) failed", file=sys.stderr)
    sys.exit(1)
print("\nall full-data checks passed")
PY
}
step "full-data-only checks" full_data_checks

echo
echo "full_check: all steps passed on $DATA"
