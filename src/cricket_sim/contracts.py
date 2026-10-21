"""Data contracts: one Open Data Contract Standard (ODCS v3.1.0) file per table, the single source of truth.

    contracts/<layer>/<table>.yaml      layer = staging | silver | gold

A contract states what one row is (grain), the key, every column's meaning, type and unit, which upstream
sources and tables it is built from, and the checks the data must pass. Everything else is generated
from it: the build's assertions (enforce), the licence lineage (lineage.py), the per-ball timing registry
(ball_layers.py), the data dictionary and lineage graph (docs.py), and the Croissant metadata (export.py).

Project conventions inside the standard (ODCS `customProperties`):
  table level   layer, kind (staging / entity / event / reference / derived / dim / fact / map / report),
                sources (upstream source ids, see lineage.SOURCE_LICENCE), upstream (tables it reads),
                timingClass (per-ball layers only)
  column level  timingClass (per-ball tables), unit, extraSources (column adds a source beyond its table's)

Checks run on every build (any failure stops the build):
  - every built table has a contract and every contract has a table
  - column names, order and physical types match exactly
  - primary key columns are unique together and never NULL; `required` columns never NULL;
    `unique` columns unique
  - every `relationships` foreign key resolves
  - every `quality` rule (ODCS library metrics, or SQL with ${table} / ${column} / ${table:other_table}) holds
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from .paths import DATA, GOLD, MODELS, ROOT, SILVER, STAGING
from .transaction import pending_publications

CONTRACTS = ROOT / "contracts"
SCHEMA_FILE = CONTRACTS / "_schema" / "odcs-json-schema-v3.1.0.json"
LAYERS = {"staging": STAGING, "silver": SILVER, "gold": GOLD, "models": MODELS}
BUILD_LAYERS = ("staging", "silver", "gold")  # `cricket build`; `cricket model` writes the models layer
TABLE_PROPERTIES = {"layer", "kind", "sources", "upstream", "timingClass"}
COLUMN_PROPERTIES = {"timingClass", "unit", "extraSources"}
KINDS = {"staging", "entity", "event", "reference", "derived", "dim", "fact", "map", "report", "model"}
OPERATORS = {
    "mustBe": lambda v, x: v == x, "mustNotBe": lambda v, x: v != x,
    "mustBeGreaterThan": lambda v, x: v > x, "mustBeGreaterOrEqualTo": lambda v, x: v >= x,
    "mustBeLessThan": lambda v, x: v < x, "mustBeLessOrEqualTo": lambda v, x: v <= x,
    "mustBeBetween": lambda v, x: x[0] <= v <= x[1], "mustNotBeBetween": lambda v, x: not x[0] <= v <= x[1],
}


def _custom(items) -> dict:
    return {c["property"]: c["value"] for c in items or []}


@dataclass(frozen=True)
class Contract:
    table: str
    layer: str
    path: Path
    doc: dict

    @property
    def obj(self) -> dict:
        return self.doc["schema"][0]

    @property
    def columns(self) -> list[dict]:
        return self.obj["properties"]

    @property
    def column_names(self) -> list[str]:
        return [c["name"] for c in self.columns]

    @property
    def props(self) -> dict:
        return _custom(self.doc.get("customProperties"))

    def column_props(self, column: dict) -> dict:
        return _custom(column.get("customProperties"))

    @property
    def primary_key(self) -> list[str]:
        keyed = [c for c in self.columns if c.get("primaryKey")]
        return [c["name"] for c in sorted(keyed, key=lambda c: c["primaryKeyPosition"])]

    @property
    def sources(self) -> list[str]:
        return list(self.props.get("sources", []))

    @property
    def data_path(self) -> Path:
        return LAYERS[self.layer] / f"{self.table}.parquet"


@cache
def load_all() -> dict[str, Contract]:
    found = {}
    for layer in LAYERS:
        for path in sorted((CONTRACTS / layer).glob("*.yaml")):
            doc = yaml.safe_load(path.read_text())
            found[path.stem] = Contract(path.stem, layer, path, doc)
    return found


def get(table: str) -> Contract:
    contracts = load_all()
    if table not in contracts:
        raise SystemExit(f"{table}: no contract in contracts/<layer>/{table}.yaml")
    return contracts[table]


def refuse_half_published(data: Path = DATA) -> None:
    """A killed publish leaves a mixed tree (new staging, old silver) that checks each table alone would pass.
    Stop until a writer command (build, model, export) has rolled it back."""
    pending = pending_publications(data)
    if pending:
        moved = "; ".join(f"{r['id'][:8]} ({', '.join(r['destinations'])})" for r in pending)
        raise SystemExit(f"{data}: half-published tree, publication {moved} never committed (killed, or a writer is "
                         "publishing right now). Run `cricket build`, `model` or `export` to roll it back, then retry.")


def lint() -> list[str]:
    """Validate every contract against the official ODCS JSON schema and the project's own rules."""
    import jsonschema

    validator = jsonschema.Draft201909Validator(json.loads(SCHEMA_FILE.read_text()))
    problems = []
    for name, c in load_all().items():
        where = c.path.relative_to(CONTRACTS.parent)
        problems += [f"{where}: {'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(c.doc)]
        if c.doc.get("id") != f"cricket-sim.{c.layer}.{name}" or c.obj.get("name") != name or len(c.doc["schema"]) != 1:
            problems.append(f"{where}: id must be cricket-sim.{c.layer}.{name} with one schema object named {name}")
        props = c.props
        if props.get("layer") != c.layer or props.get("kind") not in KINDS:
            problems.append(f"{where}: customProperties layer must be {c.layer!r} and kind one of {sorted(KINDS)}")
        if set(props) - TABLE_PROPERTIES:
            problems.append(f"{where}: unknown table customProperties {sorted(set(props) - TABLE_PROPERTIES)}")
        if not c.obj.get("description") or not c.obj.get("dataGranularityDescription"):
            problems.append(f"{where}: needs a description and a dataGranularityDescription (what one row is)")
        if not props.get("sources"):
            problems.append(f"{where}: customProperties.sources must list at least one source")
        for t in props.get("upstream", []):
            if t not in load_all():
                problems.append(f"{where}: upstream table {t!r} has no contract")
        for col in c.columns:
            extra = set(c.column_props(col)) - COLUMN_PROPERTIES
            if not col.get("description") or "TODO" in col.get("description", ""):
                problems.append(f"{where}: column {col['name']} needs a description")
            if extra:
                problems.append(f"{where}: column {col['name']} has unknown customProperties {sorted(extra)}")
        if props.get("timingClass"):
            classes = {c.column_props(col).get("timingClass") for col in c.columns if not col.get("primaryKey")}
            # `hindsight` may annotate individual columns of a pre_ball table (see ball_layers.py)
            if classes - {"hindsight"} != {props["timingClass"]}:
                problems.append(f"{where}: every non-key column must carry timingClass {props['timingClass']!r}"
                                " (or 'hindsight')")
    return problems


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rule_value(con, c: Contract, rule: dict, column: str | None) -> float:
    rel = f"read_parquet('{c.data_path}')"
    col = _qident(column) if column else None
    if rule.get("type") == "sql":
        # ${table} is this table; ${table:name} is any other contracted table, wherever the data root is
        sql = re.sub(r"\$\{table:(\w+)\}", lambda m: f"read_parquet('{load_all()[m.group(1)].data_path}')", rule["query"])
        sql = sql.replace("${table}", rel)
        return con.execute(sql.replace("${column}", col) if col else sql).fetchone()[0]
    metric, args = rule["metric"], rule.get("arguments") or {}
    if metric == "rowCount":
        return con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    if metric == "nullValues":
        return con.execute(f"SELECT count(*) FROM {rel} WHERE {col} IS NULL").fetchone()[0]
    if metric == "duplicateValues":
        return con.execute(f"SELECT count({col}) - count(DISTINCT {col}) FROM {rel}").fetchone()[0]
    if metric == "invalidValues":
        conds = []
        if "validValues" in args:
            conds.append(f"{col} NOT IN ({', '.join(repr(v) for v in args['validValues'])})")
        if "pattern" in args:
            conds.append(f"NOT regexp_full_match(CAST({col} AS VARCHAR), {args['pattern']!r})")
        return con.execute(f"SELECT count(*) FROM {rel} WHERE {col} IS NOT NULL AND ({' OR '.join(conds)})").fetchone()[0]
    raise SystemExit(f"{c.table}: unsupported quality metric {metric}")


def _check_rules(con, c: Contract, rules, column: str | None) -> tuple[int, list[str]]:
    failures = []
    for rule in rules or []:
        if rule.get("type") == "text":
            continue
        value = _rule_value(con, c, rule, column)
        op = next(k for k in OPERATORS if k in rule)
        if not OPERATORS[op](value, rule[op]):
            label = rule.get("name") or rule.get("metric") or rule.get("description", "sql rule")
            failures.append(f"{c.table}{'.' + column if column else ''}: {label} = {value}, expected {op} {rule[op]}")
    return len(rules or []), failures


def enforce(con, layers: tuple = BUILD_LAYERS) -> dict:
    """Check every built table of `layers` against its contract. Raises SystemExit listing every failure."""
    contracts = {t: c for t, c in load_all().items() if c.layer in layers}
    built = {p.stem: p for layer in layers for p in LAYERS[layer].glob("*.parquet")}
    failures = [f"{t}: built but has no contract" for t in sorted(set(built) - set(contracts))]
    failures += [f"{t}: contract exists but the table was not built" for t in sorted(set(contracts) - set(built))]
    failures += [f"{t}: built in {built[t].parent.name}, contract says {c.layer}"
                 for t, c in contracts.items() if t in built and built[t] != c.data_path]
    if failures:
        raise SystemExit("contract check failed:\n  " + "\n  ".join(failures))

    checks = 0
    for c in contracts.values():
        rel = f"read_parquet('{c.data_path}')"
        actual = [(n, t) for n, t, *_ in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
        expected = [(col["name"], col["physicalType"]) for col in c.columns]
        checks += 1
        if actual != expected:
            diff = [f"{a} != {e}" for a, e in zip(actual, expected) if a != e][:4]
            failures.append(f"{c.table}: columns differ from contract ({len(actual)} vs {len(expected)}; {diff})")
            continue
        if c.primary_key:
            key = ", ".join(map(_qident, c.primary_key))
            dupes, nulls = con.execute(
                f"SELECT count(*) - count(DISTINCT ({key})), count(*) FILTER ({' OR '.join(f'{_qident(k)} IS NULL' for k in c.primary_key)}) "
                f"FROM {rel}").fetchone()
            checks += 1
            if dupes or nulls:
                failures.append(f"{c.table}: primary key ({key}) has {dupes} duplicates and {nulls} NULLs")
        for col in c.columns:
            name = _qident(col["name"])
            if col.get("required") and not col.get("primaryKey"):
                checks += 1
                n = con.execute(f"SELECT count(*) FROM {rel} WHERE {name} IS NULL").fetchone()[0]
                if n:
                    failures.append(f"{c.table}.{col['name']}: required but {n} NULLs")
            if col.get("unique"):
                checks += 1
                n = con.execute(f"SELECT count({name}) - count(DISTINCT {name}) FROM {rel}").fetchone()[0]
                if n:
                    failures.append(f"{c.table}.{col['name']}: unique but {n} duplicates")
            for rel_def in col.get("relationships") or []:
                target_table, target_col = rel_def["to"].split(".")
                target = load_all()[target_table].data_path
                checks += 1
                n = con.execute(f"SELECT count(*) FROM {rel} s WHERE s.{name} IS NOT NULL AND s.{name} NOT IN "
                                f"(SELECT {_qident(target_col)} FROM read_parquet('{target}'))").fetchone()[0]
                if n:
                    failures.append(f"{c.table}.{col['name']}: {n} values missing from {rel_def['to']}")
            n_rules, rule_failures = _check_rules(con, c, col.get("quality"), col["name"])
            checks += n_rules
            failures += rule_failures
        n_rules, rule_failures = _check_rules(con, c, c.obj.get("quality"), None)
        checks += n_rules
        failures += rule_failures
    if failures:
        raise SystemExit("contract check failed:\n  " + "\n  ".join(failures))
    return {"tables": len(contracts), "checks_passed": checks}


def timing_registry() -> dict:
    """{table: (timingClass, [non-key columns])} for every per-ball layer table, in contract column order."""
    return {t: (c.props["timingClass"], [n for n in c.column_names if n not in c.primary_key])
            for t, c in load_all().items() if c.props.get("timingClass")}


def table_sources() -> dict:
    return {t: c.sources for t, c in load_all().items()}


def column_sources() -> dict:
    return {(t, col["name"]): c.column_props(col)["extraSources"]
            for t, c in load_all().items() for col in c.columns if c.column_props(col).get("extraSources")}
