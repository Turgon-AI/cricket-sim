"""`cricket export`: write the dataset export, enforcing licences from lineage, never from intent."""

import hashlib
import json
import re
import shutil
import sys
from urllib.parse import quote

import duckdb

from . import contracts, lineage
from .integrity import verify_release_inputs
from .transaction import new_stage, publish, writer_lock
from .paths import DATA, GOLD, MODELS, ROOT, load_yaml

CSV_MAX_ROWS = 2_000_000  # larger tables ship as Parquet only

ATTRIBUTION = {
    "ODC-BY-1.0": "Contains data from Cricsheet (https://cricsheet.org), Open Data Commons Attribution License 1.0.",
    "CC-BY-SA-4.0": "Contains data from Wikipedia season pages (CC BY-SA 4.0). This file must stay CC BY-SA 4.0.",
    "ODbL-1.0": "Contains data © OpenStreetMap contributors (ODbL 1.0) and Wikidata (CC0). This file must stay ODbL.",
    "CC-BY-4.0": "Weather data by Open-Meteo.com (CC BY 4.0), from ECMWF ERA5 reanalysis.",
    "CC0-1.0": "Public domain (CC0 1.0).",
}

# The sources whose credit the licence line above already names. Any other source in a file's lineage adds
# its own credit (the `attribution` string in config) to that file's row, so no file loses a credit it needs.
NAMED_BY_LICENCE_LINE = {
    "ODC-BY-1.0": {"cricsheet", "cricsheet_missing"},
    "CC-BY-SA-4.0": {"wiki_fixtures"},
    "ODbL-1.0": {"venues_geo", "wikidata_players"},
    "CC-BY-4.0": {"weather"},
    "CC0-1.0": set(),
}
OWN_WORK = {"derived"}  # our own computations: no third-party credit
CC_BY_SA_URI = "https://creativecommons.org/licenses/by-sa/4.0/"
WIKIPEDIA_CHANGE_NOTE = ("Licence: CC BY-SA 4.0 (" + CC_BY_SA_URI + "). Changes made: this file is modified from the "
                         "original: Wikipedia match tables were parsed, de-duplicated and linked to Cricsheet matches. "
                         "The Wikipedia pages and revisions used are listed in the 'Wikipedia sources' section below.")


# Only finished products leave the pipeline: gold marts and model outputs, never silver, staging or int_ tables.
PUBLISHABLE_PREFIXES = ("dim_", "fact_", "map_", "report_", "model_")
PUBLISHABLE_LAYERS = ("gold", "models")


def _table_path(table: str):
    if not table.startswith(PUBLISHABLE_PREFIXES):
        raise SystemExit(f"{table}: only gold dim_/fact_/map_/report_ tables and model_ outputs can be exported")
    c = contracts.get(table)
    if c.layer not in PUBLISHABLE_LAYERS:
        raise SystemExit(f"{table}: only gold dim_/fact_/map_/report_ tables and model_ outputs can be exported")
    if not c.data_path.exists():
        raise SystemExit(f"{table}: not built")
    return c.data_path


def _columns(table: str) -> list[str]:
    return [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{_table_path(table)}')").fetchall()]


def plan() -> list[dict]:
    """Validate every file in config/export.yml; raise before anything is written."""
    files = load_yaml("export.yml")["files"]
    for f in files:
        columns = None
        if f.get("drop_columns"):
            columns = [c for c in _columns(f["table"]) if c not in f["drop_columns"]]
        required = lineage.required_licence(f["table"], columns)
        if f["licence"] != required:
            raise SystemExit(f"{f['file']}: declared {f['licence']} but its sources require {required}")
        f["sources"] = lineage.source_closure(f["table"], columns)
    return files


def _source_attributions() -> dict:
    found = {}
    for name in lineage.SOURCE_CONFIGS:
        for source, cfg in (load_yaml(name) or {}).items():
            if isinstance(cfg, dict) and cfg.get("attribution"):
                found[source] = " ".join(str(cfg["attribution"]).split())
    return found


def credit_text(f: dict) -> str:
    """The credit line of one file: the licence line, plus the credit of every other source in its lineage."""
    parts = [ATTRIBUTION[f["licence"]]]
    attributions = _source_attributions()
    for source in f["sources"]:
        if source in OWN_WORK or source in NAMED_BY_LICENCE_LINE[f["licence"]]:
            continue
        if source not in attributions:
            raise SystemExit(f"{f['file']}: built from '{source}', which has no attribution in config; cannot credit it")
        text = attributions[source]
        parts.append(text if text.endswith(".") else text + ".")
    if "wiki_fixtures" in f["sources"]:
        parts.append(WIKIPEDIA_CHANGE_NOTE)
    return " ".join(parts)


def wikipedia_sources() -> list[tuple[str, int]]:
    """Every Wikipedia page and revision the CC BY-SA files were built from (from the silver snapshot)."""
    path = contracts.get("wikipedia_fixtures").data_path
    return duckdb.sql(f"SELECT DISTINCT page_title, page_revid FROM read_parquet('{path}') "
                      "WHERE page_title IS NOT NULL ORDER BY page_title, page_revid").fetchall()


def wikipedia_sources_md() -> list[str]:
    lines = ["", "## Wikipedia sources", "",
             "Pages and revisions behind the CC BY-SA 4.0 files (Wikipedia contributors, English Wikipedia, "
             f"{CC_BY_SA_URI}):", ""]
    for title, revid in wikipedia_sources():
        lines.append(f"- {title}: https://en.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'))}&oldid={revid}")
    return lines


def export(allow_dirty: bool = False) -> None:
    with writer_lock(DATA):
        run = json.loads((GOLD / "run.json").read_text())
        name = "export/" + run["build_fingerprint"][:12]
        stage = new_stage(DATA, "export")
        _write_export(allow_dirty, stage/name)
        _refuse_to_replace(DATA/name, stage/name)
        publish(DATA, stage, [name])


def _tree_digest(folder) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _refuse_to_replace(live, staged) -> None:
    """An export folder is named from the data fingerprint only. A re-export that changes any file in it (another
    commit, changed docs, contracts or export config) must not silently replace the folder people already hold."""
    if not live.exists():
        return
    old, new = _tree_digest(live), _tree_digest(staged)
    changed = sorted({n for n in old.keys() | new.keys() if old.get(n) != new.get(n)})
    if changed:
        raise SystemExit(f"{live.name}: an export with this name already exists and the new one differs in "
                         f"{len(changed)} file(s) ({', '.join(changed[:8])}{', ...' if len(changed) > 8 else ''}). "
                         f"Not replacing it. Move {live} aside (for example to ~/.Trash) and run export again. "
                         f"The new export is kept in {staged}.")


CARDS = ("DATASET_CARD.md", "MODEL_CARD.md")
CARD_STAMP = re.compile(r"<!--\s*cricket-sim card stamp:(.*?)-->", re.S)


def _fact(value, stamp="{}", shown="{}") -> tuple[str, str]:
    """(the value as written in a card's stamp, the way the card's prose must show it)."""
    return ("none", "no value") if value is None else (stamp.format(value), shown.format(value))


def card_facts(run: dict, model_run: dict) -> dict:
    """What each card must say about the build it ships with: the fingerprints and the headline numbers."""
    build, model = run["build_fingerprint"][:16], model_run["model_fingerprint"][:16]
    headline = json.loads((GOLD / "headline.json").read_text())
    con = duckdb.connect()
    afghan = con.execute(f"SELECT count(*) FROM read_parquet('{GOLD / 'report_international_coverage.parquet'}') "
                         "WHERE segment = 'ODI' AND status = 'afghanistan_withheld_by_cricsheet'").fetchone()[0]
    brier = dict(con.execute(f"SELECT slice, brier FROM read_parquet('{MODELS / 'report_model_scores.parquet'}') "
                             "WHERE target = 'win_probability' AND model = 'model' AND scope = 'all'").fetchall())
    return {
        "DATASET_CARD.md": {
            "build_fingerprint": _fact(build),
            "matches_in_scope": _fact(headline.get("matches_in_scope"), "{}", "{:,}"),
            "balls_in_scope": _fact(headline.get("balls_in_scope"), "{}", "{:,}"),
            "pct_matches_with_weather": _fact(headline.get("pct_matches_with_weather"), "{:g}", "{:g}%"),
            "pct_in_scope_players_with_dob_wikidata": _fact(headline.get("pct_in_scope_players_with_dob_wikidata"),
                                                            "{:g}", "{:g}%"),
            "afghanistan_withheld_odis": _fact(afghan, "{}", "{} involve Afghanistan"),
        },
        "MODEL_CARD.md": {
            "build_fingerprint": _fact(build),
            "model_fingerprint": _fact(model),
            "innings_1_brier": _fact(brier.get("innings_1"), "{:.4f}", "{:.4f}"),
            "innings_2_brier": _fact(brier.get("innings_2"), "{:.4f}", "{:.4f}"),
        },
    }


def refresh_stamp(text: str, facts: dict) -> str:
    """The card text with its stamp line rewritten from this build's facts (added at the end if missing).
    Only the stamp: the card's prose still has to show the numbers, or the export check refuses it."""
    stamp = "<!-- cricket-sim card stamp: " + " ".join(f"{k}={v[0]}" for k, v in facts.items()) + " -->"
    if CARD_STAMP.search(text):
        return CARD_STAMP.sub(lambda _: stamp, text, count=1)
    return text.rstrip("\n") + "\n\n" + stamp + "\n"


def card_problems(card: str, text: str, facts: dict) -> list[str]:
    """Where a card disagrees with the build it would ship with. The card carries a one-line stamp of the build
    fingerprint and headline numbers it was written for; each stamped value must match, and show in the prose."""
    found = CARD_STAMP.search(text)
    if not found:
        return [f"{card}: has no 'cricket-sim card stamp' line, so it cannot be tied to this build"]
    stamp = dict(pair.split("=", 1) for pair in found.group(1).split() if "=" in pair)
    prose = text.replace(found.group(0), "")
    problems = []
    for key, (value, shown) in facts.items():
        if stamp.get(key) != value:
            problems.append(f"{card}: stamp {key} is {stamp.get(key, 'missing')}, this build has {value}")
        elif shown not in prose:
            problems.append(f"{card}: stamp {key}={value} but the text does not show '{shown}'")
    return problems


def _checked_cards(run: dict, model_run: dict, allow_dirty: bool) -> dict:
    """The card texts to ship. A card that does not match this build is a refusal. With --allow-dirty (tests
    only, never a release) it ships with a warning at the top instead, so it cannot pass for this build's card."""
    facts = card_facts(run, model_run)
    texts, problems = {}, {}
    for card in CARDS:
        texts[card] = (ROOT / "docs" / card).read_text()
        problems[card] = card_problems(card, texts[card], facts[card])
    every = [p for card in CARDS for p in problems[card]]
    if every and not allow_dirty:
        raise SystemExit("the cards in docs/ do not match this build and model; update them from the current "
                         "numbers, then export again:\n  " + "\n  ".join(every))
    for card in CARDS:
        if problems[card]:
            print(f"WARNING: {card} does not match this build; shipping it with a warning ({len(problems[card])} mismatches)",
                  file=sys.stderr)
            texts[card] = ("> **WARNING: this card does not describe this export.** It was written for another build "
                           "(export made with --allow-dirty, tests only; not a release).\n> Mismatches:\n"
                           + "".join(f"> - {p}\n" for p in problems[card]) + "\n" + texts[card])
    return texts


def _write_export(allow_dirty: bool, out) -> None:
    files = plan()
    run = json.loads((GOLD / "run.json").read_text())
    model_run = verify_release_inputs(DATA, run, allow_dirty)
    cards = _checked_cards(run, model_run, allow_dirty)
    con = duckdb.connect()
    problems = contracts.lint()
    if problems:
        raise SystemExit("contracts do not lint: " + "; ".join(problems))
    contracts.enforce(con)
    contracts.enforce(con, layers=("models",))
    out.mkdir(parents=True)

    manifest, dictionary = [], []
    for f in files:
        src = _table_path(f["table"])
        select = "* EXCLUDE (" + ", ".join(f["drop_columns"]) + ")" if f.get("drop_columns") else "*"
        sql = f"SELECT {select} FROM read_parquet('{src}')"  # gold marts are already in scope
        rows = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
        con.execute(f"COPY ({sql}) TO '{out / (f['file'] + '.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
        written = [f["file"] + ".parquet"]
        if rows <= CSV_MAX_ROWS:
            con.execute(f"COPY ({sql}) TO '{out / (f['file'] + '.csv')}' (HEADER)")
            written.append(f["file"] + ".csv")
        columns = con.execute(f"DESCRIBE {sql}").fetchall()
        dictionary.append((f, rows, columns))
        for name in written:
            manifest.append({"file": name, "licence": f["licence"], "rows": rows,
                             "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()})

    title = load_yaml("export.yml")["title"]
    lines = [f"# {title}", "",
             f"Built from pipeline fingerprint `{run['build_fingerprint'][:16]}` (commit `{(run['code']['commit'] or 'unknown')[:10]}`).",
             "Each file carries its own licence; see LICENSES.md. Machine-readable metadata: croissant.json.", ""]
    for f, rows, _ in dictionary:
        c = contracts.get(f["table"])
        lines += [f"## {f['file']}  ({rows:,} rows, {f['licence']})", "", c.obj["description"].strip(), "",
                  f"One row is: {c.obj['dataGranularityDescription'].strip()} Key: {', '.join(c.primary_key)}.", "",
                  "| column | type | description |", "|---|---|---|"]
        for col in c.columns:
            unit = c.column_props(col).get("unit")
            desc = " ".join(col["description"].split()) + (f" ({unit})" if unit else "")
            lines.append(f"| `{col['name']}` | {col['physicalType']} | {desc} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines))
    licences = ["# Licences", "", "| file | licence | attribution |", "|---|---|---|"]
    credit = {f["file"]: credit_text(f) for f in files}
    licences += [f"| {m['file']} | {m['licence']} | {credit[m['file'].rsplit('.', 1)[0]]} |" for m in manifest]
    if any("wiki_fixtures" in f["sources"] for f in files):
        licences += wikipedia_sources_md()
    (out / "LICENSES.md").write_text("\n".join(licences) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "croissant.json").write_text(json.dumps(croissant(title, run, files, manifest), indent=2) + "\n")
    # the dataset card and the attribution notice travel with the data
    for card, text in cards.items():
        (out / card).write_text(text)
    shutil.copy(ROOT / "NOTICE", out / "NOTICE")
    print(f"exported {len(files)} tables to {out}")


CROISSANT_TYPES = {"string": "sc:Text", "integer": "sc:Integer", "number": "sc:Float", "boolean": "sc:Boolean",
                   "date": "sc:Date", "timestamp": "sc:DateTime", "array": "sc:Text", "object": "sc:Text"}
SPDX = {"ODC-BY-1.0": "https://spdx.org/licenses/ODC-By-1.0.html", "CC-BY-SA-4.0": "https://spdx.org/licenses/CC-BY-SA-4.0.html",
        "ODbL-1.0": "https://spdx.org/licenses/ODbL-1.0.html", "CC-BY-4.0": "https://spdx.org/licenses/CC-BY-4.0.html",
        "CC0-1.0": "https://spdx.org/licenses/CC0-1.0.html"}


def croissant(title: str, run: dict, files: list[dict], manifest: list[dict]) -> dict:
    """MLCommons Croissant 1.0 metadata, generated from the contracts (one RecordSet per Parquet file)."""
    sha = {m["file"]: m["sha256"] for m in manifest}
    distribution, record_sets = [], []
    for f in files:
        c, name = contracts.get(f["table"]), f["file"] + ".parquet"
        distribution.append({"@type": "cr:FileObject", "@id": name, "name": name, "contentUrl": name,
                             "encodingFormat": "application/x-parquet", "sha256": sha[name],
                             "description": f"Licence: {f['licence']}"})
        record_sets.append({
            "@type": "cr:RecordSet", "@id": f["file"], "name": f["file"],
            "description": " ".join(c.obj["description"].split()),
            "key": [{"@id": f"{f['file']}/{k}"} for k in c.primary_key],
            "field": [{"@type": "cr:Field", "@id": f"{f['file']}/{col['name']}", "name": col["name"],
                       "description": " ".join(col["description"].split()),
                       "dataType": CROISSANT_TYPES[col["logicalType"]],
                       "source": {"fileObject": {"@id": name}, "extract": {"column": col["name"]}}}
                      for col in c.columns],
        })
    return {
        "@context": {"@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
                     "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
                     "dct": "http://purl.org/dc/terms/", "citeAs": "cr:citeAs", "column": "cr:column",
                     "conformsTo": "dct:conformsTo", "data": {"@id": "cr:data", "@type": "@json"},
                     "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "examples": {"@id": "cr:examples", "@type": "@json"},
                     "extract": "cr:extract", "field": "cr:field", "fileObject": "cr:fileObject", "fileSet": "cr:fileSet",
                     "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
                     "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField",
                     "path": "cr:path", "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
                     "repeated": "cr:repeated", "replace": "cr:replace", "separator": "cr:separator", "source": "cr:source",
                     "subField": "cr:subField", "transform": "cr:transform"},
        "@type": "sc:Dataset", "conformsTo": "http://mlcommons.org/croissant/1.0", "name": title,
        "description": "Men's international and IPL ball-by-ball cricket from Cricsheet, with the match state before "
                       "every ball, venue, weather and player context, and per-match quality checks.",
        "license": sorted({SPDX[f["licence"]] for f in files}),
        "version": run["build_fingerprint"][:12],
        "citeAs": "Cricsheet (https://cricsheet.org), ODC-BY 1.0; see LICENSES.md for every source.",
        "distribution": distribution, "recordSet": record_sets,
    }
