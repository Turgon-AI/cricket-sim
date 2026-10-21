"""Make the tiny committed fixture (tests/fixtures/tiny/) that the end-to-end test and CI build from.

    uv run python scripts/make_fixture.py          # needs the real bronze snapshots in data/bronze

Ten real matches, each chosen to exercise something the pipeline must get right (see MATCHES), plus
every openly licensed source they need (Cricsheet, Wikipedia, Wikidata, OpenStreetMap, Open-Meteo),
copied or subset from the locked real snapshots.

Ari's Cricinfo-derived data and the HawkeyeStats data may NOT be redistributed, so the fixture holds
SYNTHETIC stand-ins in exactly their raw formats: same columns, same types as DuckDB infers them, the
same sentinels ('NA', -1, (5, 5)), aligned to the real Cricsheet deliveries so the joins and the
Hawk-Eye link are exercised. Every value in them is generated here from a fixed seed; none is real.

Deterministic: the same real snapshots give byte-identical fixture files.
"""

import csv
import io
import json
import random
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cricket_sim import bronze  # noqa: E402
from cricket_sim.venues import store  # noqa: E402

REAL = ROOT / "data" / "bronze"
OUT = ROOT / "tests" / "fixtures" / "tiny"
FROZEN_TIME = "2026-10-16T00:00:00Z"

MATCHES = {
    "433606": "2011 World Cup final, India v Sri Lanka (ODI)",
    "1144528": "2019 World Cup semi-final, India v New Zealand: Dhoni's run-out (ODI, Hawk-Eye linked)",
    "1144530": "2019 World Cup final, England v New Zealand: tie, super over, boundary count (ODI)",
    "1384439": "2023 World Cup final, India v Australia (ODI)",
    "1022353": "2017 Champions Trophy, India v Pakistan: rain-revised chase (ODI, D/L)",
    "1389401": "Rajkot 2024, India v England: 5 penalty runs to start England's innings (Test)",
    "1223872": "Gabba 2021, Australia v India: fourth-innings chase (Test)",
    "1187669": "Eden Park 2019, New Zealand v England: decided by a super over (T20I)",
    "1181768": "2019 IPL final, Mumbai Indians v Chennai Super Kings (IPL, Hawk-Eye linked)",
    "1085975": "2017 Women's World Cup final, England v India: out of scope, must be filtered out",
}
HAWKEYE = {"1144528": ("mensODI", 9001), "1181768": ("mensIPL", 9002)}


def lock() -> dict:
    return yaml.safe_load((ROOT / "config" / "snapshots.lock.yml").read_text())


def real(source: str) -> Path:
    return REAL / source / lock()[source]


def out(source: str) -> Path:
    folder = OUT / "bronze" / source / lock()[source]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_manifest(source: str, files: list[dict], upstream: dict) -> None:
    folder = out(source)
    for e in files:
        p = folder / e["stored_as"]
        e["stored_bytes"] = p.stat().st_size
        e["stored_sha256"] = bronze.sha256(p)
    manifest = {"source": source, "snapshot_id": lock()[source], "ingested_at_utc": FROZEN_TIME,
                "upstream": upstream, "files": sorted(files, key=lambda e: e["stored_as"])}
    (folder / "_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")


def store_zst(source: str, name: str, data: bytes) -> dict:
    raw = out(source) / name
    raw.write_bytes(data)
    digest = bronze.compress_to_zst(raw, out(source) / f"{name}.zst")
    raw.unlink()
    return {"name": name, "stored_as": f"{name}.zst", "compressed": True, "raw_bytes": len(data), "raw_sha256": digest}


def copy_verbatim(source: str) -> None:
    shutil.copytree(real(source), out(source), dirs_exist_ok=True)


# ---------------------------------------------------------------- Cricsheet (real subset, ODC-BY)
def cricsheet() -> dict[str, dict]:
    docs, people = {}, set()
    with zipfile.ZipFile(real("cricsheet") / "all_json.zip") as z:
        for mid in MATCHES:
            docs[mid] = json.loads(z.read(f"{mid}.json"))
            people |= set(docs[mid]["info"]["registry"]["people"].values())
        licence = z.read("LICENSE.txt")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in [("LICENSE.txt", licence)] + [(f"{m}.json", json.dumps(docs[m]).encode()) for m in sorted(docs)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    (out("cricsheet") / "all_json.zip").write_bytes(buf.getvalue())
    files = [{"name": "all_json.zip", "stored_as": "all_json.zip", "compressed": False}]
    for name in ("people.csv", "names.csv"):
        lines = (real("cricsheet") / name).read_text().splitlines(keepends=True)
        keep = [lines[0]] + [ln for ln in lines[1:] if ln.split(",", 1)[0] in people]
        (out("cricsheet") / name).write_text("".join(keep))
        files.append({"name": name, "stored_as": name, "compressed": False})
    write_manifest("cricsheet", files, {"fixture_subset_of": lock()["cricsheet"], "matches": sorted(MATCHES)})
    return docs


def deliveries(doc: dict):
    """(innings, over, ball_in_over, delivery_number, delivery dict) in Cricsheet order, super overs included."""
    for i, inn in enumerate(doc["innings"], start=1):
        n = 0
        for ov in inn.get("overs", []):
            for k, d in enumerate(ov["deliveries"], start=1):
                n += 1
                yield i, ov["over"], k, n, d


# ---------------------------------------------------------------- synthetic Ari (format only)
# Category labels as the real formats spell them (labels only; no real per-ball value is copied)
LINES = ["outside offstump", "on the stumps", "wide outside offstump", "down leg", "wide down leg"]
LENGTHS = ["good length", "full", "short of a good length", "short", "full toss", "yorker"]
SHOTS = ["defended", "flick", "on drive", "forward defence", "leg glance", "cover drive", "left alone", "cut shot", "pull"]
CONTROL = ["in control", "not in control"]
DIRECTIONS = ["cover", "mid wicket", "mid off", "mid on", "point", "square leg", "third man", "fine leg"]
HAWKEYE_STYLES = ["FAST_SEAM", "MEDIUM_SEAM", "OFF_SPIN", "LEG_SPIN", "ORTHODOX"]


def synthetic_ari(docs: dict) -> None:
    rnd = random.Random(20261016)
    tracking = [["match_id", "innings", "delivery_number", "ball_line", "ball_length", "shot_played", "shot_control",
                 "wagon_x", "wagon_y", "wagon_zone", "wagon_angle", "wagon_distance", "wagon_in_ring", "shot_direction",
                 "ball_timestamp", "win_probability", "predicted_score", "ball_events"]]
    preds = [["match_id", "innings", "delivery_number"] + [f"{f}{s}" for f in ("line", "length", "shot", "direction")
                                                            for s in ("", "_confidence", "_explicit")]]
    for mid in sorted(docs, key=int):
        doc = docs[mid]
        start = datetime.fromisoformat(doc["info"]["dates"][0]) + timedelta(hours=9)
        wp = 50.0
        for b, (inn, _, _, n, d) in enumerate(deliveries(doc)):
            wicket = bool(d.get("wickets"))
            wp = min(95.0, max(5.0, wp + rnd.uniform(-0.4, 0.4) - (5.0 if wicket else 0.0)))
            labelled = rnd.random() < 0.8
            ts = (start + timedelta(seconds=40 * b)).strftime("%Y-%m-%d %H:%M:%S")
            tracking.append([mid, inn, n,
                             rnd.choice(LINES) if labelled else "", rnd.choice(LENGTHS) if labelled else "",
                             rnd.choice(SHOTS) if labelled else "", rnd.choice(CONTROL) if labelled else "",
                             rnd.randint(0, 360), rnd.randint(0, 360), rnd.randint(1, 8), round(rnd.uniform(0, 359.9), 1),
                             round(rnd.uniform(0.001, 1.4), 3), rnd.choice(["true", "false"]), rnd.choice(DIRECTIONS), ts,
                             round(wp, 2), rnd.randint(150, 350),
                             # DRS events where Cricsheet records a review, so the alignment check has teeth in CI
                             "drs review (unsuccessful)" if d.get("review") else ("" if rnd.random() < 0.9 else "bowler change")])
            if rnd.random() < 0.3:
                preds.append([mid, inn, n] + [x for f in (LINES, LENGTHS, SHOTS, DIRECTIONS)
                                              for x in (rnd.choice(f), round(rnd.uniform(0.3, 1.0), 3), rnd.randint(0, 1))])
    players = [["player_id", "name", "cricinfo_id", "full name", "gender", "country", "batting style", "bowling style",
                "playing role", "fielding style", "first international year", "last international year", "major teams",
                "image url"]]
    people = list(csv.DictReader(io.StringIO((out("cricsheet") / "people.csv").read_text())))
    for k, p in enumerate(people):
        if not p.get("key_cricinfo"):
            continue
        players.append([p["identifier"], p["name"], p["key_cricinfo"], f"Synthetic Player {k}", "male", "Synthetica",
                        rnd.choice(["right-hand bat", "left-hand bat"]),
                        rnd.choice(["right-arm medium", "right-arm offbreak", "legbreak", ""]),
                        rnd.choice(["batter", "bowler", "allrounder"]), "", 2005, 2020, "Synthetic XI", ""])

    def as_csv(rows):
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        return buf.getvalue().encode()

    files = [store_zst("ari", "cricinfo_tracking.csv", as_csv(tracking)),
             store_zst("ari", "cricinfo_text_predictions.csv", as_csv(preds)),
             store_zst("ari", "players.csv", as_csv(players))]
    write_manifest("ari", files, {"synthetic": True, "note": "format-only stand-in generated by scripts/make_fixture.py"})


# ---------------------------------------------------------------- synthetic Hawk-Eye (format only)
HAWKEYE_COLS = ["matchId", "delivery", "ball", "batter", "batterId", "rightHandedBat", "nonStriker", "nonStrikerId",
                "bowler", "bowlerId", "rightArmedBowl", "bowlingStyle", "ballSpeed", "dismissalDetails", "runs",
                "batterRuns", "bowlerRuns", "extras", "pitchX", "pitchY", "stumpsX", "stumpsY", "fieldX", "fieldY"]


def synthetic_hawkeye(docs: dict) -> None:
    rnd = random.Random(7)
    people = {p["identifier"]: p for p in csv.DictReader(io.StringIO((out("cricsheet") / "people.csv").read_text()))}
    by_comp = {}
    for mid, (comp, he_id) in HAWKEYE.items():
        reg = docs[mid]["info"]["registry"]["people"]
        pulse = lambda name: people.get(reg.get(name), {}).get("key_pulse") or ""  # noqa: E731
        legal = {}
        for inn, over, k, _, d in deliveries(docs[mid]):
            if rnd.random() < 0.03:
                continue  # a delivery Hawk-Eye never recorded
            extras = d.get("extras", {})
            key = (inn, over)
            legal[key] = legal.get(key, 0) + (0 if ("wides" in extras or "noballs" in extras) else 1)
            tracked = rnd.random() < 0.85
            na = lambda v: f"{v:.3f}" if tracked else "NA"  # noqa: E731
            field = (f"{rnd.uniform(5, 95):.1f}", f"{rnd.uniform(5, 95):.1f}") if tracked else rnd.choice([("5", "5"), ("NA", "NA")])
            by_comp.setdefault(comp, []).append([
                he_id, f"{inn}.{over + 1}.{k}", legal[key] or 1, d["batter"], pulse(d["batter"]),
                rnd.choice(["true", "false", "NA"]), d["non_striker"], pulse(d["non_striker"]), d["bowler"],
                pulse(d["bowler"]), rnd.choice(["true", "false", "NA"]), rnd.choice(HAWKEYE_STYLES),
                round(rnd.uniform(30, 42), 2) if tracked else -1, "", str(d["runs"]["total"]), d["runs"]["batter"],
                d["runs"]["total"], "Wd" if "wides" in extras else ("Nb" if "noballs" in extras else ""),
                na(rnd.uniform(-1.2, 1.2)), na(rnd.uniform(1, 12)), na(rnd.uniform(-0.5, 0.5)),
                na(rnd.uniform(0.1, 1.2)), *field])
    # a Hawk-Eye match with no Cricsheet counterpart, holding the one corrupt `runs` value the real data has
    by_comp["mensODI"].append([9999, "1.1.1", 1, "Nobody Real", 1, "NA", "Nobody Else", 2, "Nobody Bowls", 3, "NA",
                               "NA", -1, "", "x", 0, 0, "", "NA", "NA", "NA", "NA", "NA", "NA"])
    files = []
    for comp in ("mensIPL", "mensODI", "mensTest"):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(HAWKEYE_COLS)
        w.writerows(by_comp.get(comp, [[1, "1.1.1", 1, "A", 1, "NA", "B", 2, "C", 3, "NA", "NA", -1, "", "x", 0,
                                         0, "", "NA", "NA", "NA", "NA", "NA", "NA"]]))
        files.append(store_zst("hawkeye", f"{comp}HawkeyeStats.csv", buf.getvalue().encode()))
    write_manifest("hawkeye", files, {"synthetic": True, "note": "format-only stand-in generated by scripts/make_fixture.py"})


# ---------------------------------------------------------------- weather (real subset, CC BY 4.0)
def weather(docs: dict) -> None:
    src, dst = real("weather"), out("weather")
    pairs = {(d["info"]["venue"], d["info"].get("city")) for d in docs.values()}
    days = {(d["info"]["venue"], d["info"].get("city"), day) for d in docs.values() for day in d["info"]["dates"]}
    plan = json.loads((src / "plan.json").read_text())

    def wanted(f):
        aliases = {tuple(a) for a in f["aliases"]}
        if not aliases & pairs:
            return False
        return any((v, c, day) in days for v, c in aliases
                   for day in _dates(f["start"], f["end"]))

    plan["files"] = [f for f in plan["files"] if wanted(f)]
    plan["deferred"] = [f for f in plan.get("deferred", []) if {tuple(a) for a in f["aliases"]} & pairs]
    # keep only the fixture's venue strings, so the plan matches the fixture's matches exactly
    for f in plan["files"]:
        f["aliases"] = [a for a in f["aliases"] if tuple(a) in pairs]
    keep = {f["file"] for f in plan["files"]}
    manifest = json.loads((src / "_snapshot.json").read_text())
    entries = []
    for e in manifest["files"]:
        if e["stored_as"] in keep:
            shutil.copy2(src / e["stored_as"], dst / e["stored_as"])
            entries.append({k: v for k, v in e.items() if k not in ("stored_bytes", "stored_sha256")})
    store.write_json_atomic(dst / "plan.json", plan)
    store.write_json_atomic(dst / "failures.json", {})
    for name in ("plan.json", "failures.json"):
        entries.append({"name": name, "stored_as": name, "compressed": False})
    write_manifest("weather", entries, manifest["upstream"])


def _dates(start: str, end: str) -> list[str]:
    d, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return [(d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - d).days + 1)]


def write_lock() -> None:
    sources = sorted(lock())
    (OUT / "snapshots.lock.yml").write_text("# Fixture snapshots (scripts/make_fixture.py)\n"
                                            + yaml.safe_dump({s: lock()[s] for s in sources}, sort_keys=True))
    pins = {s: bronze.sha256(OUT / "bronze" / s / lock()[s] / "_snapshot.json") for s in sources}
    (OUT / "snapshots.pins.yml").write_text("# sha256 of each fixture snapshot's _snapshot.json\n"
                                            + yaml.safe_dump(pins, sort_keys=True))


def main() -> None:
    shutil.rmtree(OUT / "bronze", ignore_errors=True)
    docs = cricsheet()
    for source in ("cricsheet_missing", "wiki_fixtures", "wikidata_players", "venues_geo"):
        copy_verbatim(source)
    synthetic_ari(docs)
    synthetic_hawkeye(docs)
    weather(docs)
    write_lock()
    (OUT / "SUBSET").write_text("This folder holds a SUBSET of the real snapshots (scripts/make_fixture.py).\n"
                                "Checks that only hold for complete data, such as stale venue overrides, are skipped.\n")
    (OUT / "MATCHES.json").write_text(json.dumps(MATCHES, indent=2) + "\n")
    print(f"fixture written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
