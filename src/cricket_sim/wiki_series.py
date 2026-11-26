"""Scheduled start time and day/night status of limited-overs matches, from Wikipedia series pages.

Bronze source `wiki_series`: wikitext of every series page that the pinned season pages (source `wiki_fixtures`)
link with {{main|...}} from a section holding a numbered men's ODI or T20I, plus every "YYYY Indian Premier League"
season page, plus one more level of {{main|...}} sub-pages whose title starts with the page's own title (for example
"2015 Cricket World Cup Pool A" under "2015 Cricket World Cup"). Pinned by revision id.

Each match on those pages is one {{Single-innings cricket match}} / {{Limited overs international}} /
{{Limited overs matches}} template. Its `report` link holds the ESPNcricinfo match id, which is Cricsheet's match id
for these matches, its `time` field the scheduled start ("The scheduled time for the start of the match"; usually
local time, but IPL pages give Indian time even for matches played in the UAE or South Africa, and a few fields name
GMT or IST), and `daynight` / `night` (or a "D/N" mark in the time) the day/night status.

Silver: data/silver/wikipedia_match_schedule.parquet, one row per Cricinfo match id.
Licence CC BY-SA 4.0 (see config/wiki_sources.yml).
"""

import hashlib
import json
import re
import tempfile
import time as _time
from datetime import datetime, time, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import requests

from . import bronze
from .paths import SILVER, load_yaml, snapshot_dir
from .wiki_fixtures import HEADING_RE, NUM_RE, _fetch_page, _split_top, clean_text, cricinfo_id

SOURCE = "wiki_series"
INDEX_NAME = "index.json"
MAIN_RE = re.compile(r"\{\{\s*main(?:\s+article)?\s*\|([^{}]*)\}\}", re.I)
TEMPLATE_RE = re.compile(r"\{\{\s*(single[ _-]innings[ _]cricket[ _]match|limited[ _-]overs[ _]international|"
                         r"limited[ _-]overs[ _]matches)\s*(?=[|}\n])", re.I)
TIME_RE = re.compile(r"(?<![\d:])(\d{1,2})\s*[:.]\s*(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?![\d])", re.I)
NOT_MATCHES_RE = re.compile(r"squad|official|ceremon|statistic|controvers|broadcast|qualif|marketing|prize|records", re.I)
# a zone the time field names; anything else is read as local time at the ground
STATED_ZONE_RE = [(re.compile(r"\b(?:GMT|UTC)\b(?!\s*[+-−]\s*\d)"), "UTC"), (re.compile(r"\bIST\b"), "Asia/Kolkata")]
DN_RE = re.compile(r"\bD\s*/\s*N\b|day[\s/-]*night", re.I)
TRUE_WORDS = {"y", "yes", "true", "1", "d/n", "dn", "day/night", "day-night", "night"}


# --------------------------------------------------------------------------------------------- page list

def main_targets(text: str) -> list[str]:
    """Positional page titles of one {{main|A|B|l1=...}} template body."""
    out = []
    for part in text.split("|"):
        part = part.strip()
        if part and "=" not in part:
            out.append(part.replace("_", " "))
    return out


def season_series_titles(wikitext: str) -> list[str]:
    """Series pages a season page links with {{main|...}} from a section holding a numbered men's ODI or T20I.
    {{main}} applies to its own heading level and below (a level-2 main covers the level-3 sections inside it)."""
    mains: dict[int, list[str]] = {}
    level, out = 0, []
    for line in wikitext.split("\n"):
        stripped = line.strip()
        hm = HEADING_RE.match(stripped)
        if hm:
            level = len(hm.group(1))
            mains = {k: v for k, v in mains.items() if k < level}
            continue
        for m in MAIN_RE.finditer(stripped):
            mains[level] = main_targets(m.group(1))
        if mains and re.search(r"\{\{\s*crw\s*\|", stripped, re.I) is None:
            found = NUM_RE.search(clean_text(stripped))
            if found and found.group(1) in ("ODI", "T20I"):
                for title in mains[max(mains)]:
                    if title not in out:
                        out.append(title)
    return out


def ipl_titles(first_year: int, last_year: int) -> list[str]:
    return [f"{y} Indian Premier League" for y in range(first_year, last_year + 1)]


def sub_page_titles(title: str, wikitext: str) -> list[str]:
    """{{main|...}} targets on a series page that are its own sub-pages (title starts with the page's title)."""
    out = []
    for m in MAIN_RE.finditer(wikitext):
        for t in main_targets(m.group(1)):
            if t != title and t.startswith(title) and not NOT_MATCHES_RE.search(t[len(title):]) and t not in out:
                out.append(t)
    return out


def file_name(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80]
    return f"{slug}__{hashlib.sha1(title.encode()).hexdigest()[:8]}.wiki"


def ingest_wiki_series() -> str:
    """Fetch the series pages linked from the LOCKED wiki_fixtures snapshot (so the page list is reproducible),
    the IPL season pages, and their sub-pages. Pages that do not exist are recorded in the index, not fatal."""
    from .ingest import _finish, _start

    cfg = load_yaml("wiki_sources.yml")[SOURCE]
    lock = bronze.read_lock()
    if "wiki_fixtures" not in lock:
        raise SystemExit("wiki_series reads the page list from the locked wiki_fixtures snapshot; ingest that first")
    fixtures = snapshot_dir("wiki_fixtures", lock["wiki_fixtures"])
    titles: list[tuple[str, str]] = []  # (title, why)
    for entry in json.loads((fixtures / "index.json").read_text(encoding="utf-8")):
        text = bronze.read_zst(fixtures / entry["stored_as"]).decode("utf-8")
        for t in season_series_titles(text):
            if t not in {x for x, _ in titles}:
                titles.append((t, f"main link on {entry['title']}"))
    for t in ipl_titles(cfg["ipl_first_year"], cfg["ipl_last_year"]):
        titles.append((t, "IPL season page"))

    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = _start(SOURCE, snapshot_id)
    if folder is None:
        return snapshot_id
    session = requests.Session()
    session.headers["User-Agent"] = cfg["user_agent"]
    fetched, missing, seen = [], [], set()
    try:
        queue = [(t, why, 1) for t, why in titles]
        while queue:
            title, why, depth = queue.pop(0)
            if title in seen:
                continue
            seen.add(title)
            page = _fetch_page(session, cfg["endpoint"], title)
            _time.sleep(cfg.get("delay_seconds", 1.0))
            if page is None:
                missing.append({"title": title, "why": why})
                continue
            if page["resolved_title"] in seen and page["resolved_title"] != title:
                continue  # a redirect to a page already held
            seen.add(page["resolved_title"])
            page.update(why=why, depth=depth)
            fetched.append(page)
            if depth == 1:
                queue += [(t, f"sub-page of {page['resolved_title']}", 2)
                          for t in sub_page_titles(page["resolved_title"], page["wikitext"])]
            if len(fetched) % 50 == 0:
                print(f"  wiki_series: {len(fetched)} pages fetched, {len(queue)} queued")
    except BaseException:
        bronze.discard_partial(folder)
        raise

    files, index = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for page in fetched:
            name = file_name(page["resolved_title"])
            raw = Path(tmp) / name
            raw.write_text(page["wikitext"], encoding="utf-8")
            raw_sha = bronze.compress_to_zst(raw, folder / f"{name}.zst")
            files.append({"name": name, "title": page["resolved_title"], "revid": page["revid"],
                          "url": f"https://en.wikipedia.org/w/index.php?oldid={page['revid']}",
                          "stored_as": f"{name}.zst", "compressed": True,
                          "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha})
            index.append({k: page[k] for k in ("title", "resolved_title", "pageid", "revid", "parentid", "timestamp",
                                               "why", "depth")} | {"stored_as": f"{name}.zst"})
    (folder / INDEX_NAME).write_text(json.dumps({"pages": index, "missing": missing}, indent=2, ensure_ascii=False)
                                     + "\n", encoding="utf-8")
    files.append({"name": INDEX_NAME, "stored_as": INDEX_NAME, "compressed": False})
    upstream = {"endpoint": cfg["endpoint"], "pages": len(fetched), "missing_pages": len(missing),
                "page_list_from": f"wiki_fixtures/{lock['wiki_fixtures']}", "licence": cfg["licence"]}
    print(f"  wiki_series: {len(fetched)} pages, {len(missing)} titles do not exist")
    return _finish(SOURCE, snapshot_id, upstream, files)


# --------------------------------------------------------------------------------------------- parse

def template_bodies(wikitext: str):
    """Yield (template name, body) for every match template, the body being the text between the name and the
    matching closing braces."""
    for m in TEMPLATE_RE.finditer(wikitext):
        depth, i = 1, m.end()
        while i < len(wikitext) and depth:
            two = wikitext[i:i + 2]
            if two == "{{":
                depth += 1; i += 2; continue
            if two == "}}":
                depth -= 1; i += 2; continue
            i += 1
        if depth == 0:
            yield m.group(1).lower().replace("_", " ").replace("-", " "), wikitext[m.end():i - 2]


def template_fields(body: str) -> dict[str, str]:
    out = {}
    for part in _split_top(body, "|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip().lower()] = value.strip()
    return out


def parse_time(text: str) -> time | None:
    """First clock time in a cleaned time field ('14:30', '2:30 pm', '10.00 (D/N)'). None if absent or invalid."""
    m = TIME_RE.search(text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower().replace(".", "")
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def stated_zone(time_text: str) -> str | None:
    """IANA zone the time field names ('14:00 GMT' -> 'UTC', '19:30 IST' -> 'Asia/Kolkata'), else None."""
    for pattern, zone in STATED_ZONE_RE:
        if pattern.search(time_text):
            return zone
    return None


def parse_day_night(fields: dict[str, str], time_text: str) -> bool | None:
    """TRUE when the template marks a day/night match (daynight / night field, or 'D/N' in the time). A template
    that has the field but leaves it blank says nothing; it is not read as a day match."""
    for key in ("daynight", "night", "dn"):
        value = clean_text(fields.get(key, "")).strip().lower()
        if value:
            return value in TRUE_WORDS
    return True if DN_RE.search(time_text) else None


def parse_page(title: str, wikitext: str) -> list[dict]:
    rows = []
    for name, body in template_bodies(wikitext):
        f = template_fields(body)
        match_id = cricinfo_id(f.get("report", ""))
        if match_id is None:
            continue
        time_text = clean_text(f.get("time", ""))
        rows.append({"cricinfo_match_id": match_id, "template": name, "date_text": clean_text(f.get("date", "")),
                     "time_text": time_text or None, "scheduled_start_stated": parse_time(time_text),
                     "stated_time_zone": stated_zone(time_text),
                     "day_night_flag": parse_day_night(f, time_text), "page_title": title})
    return rows


SCHEMA = pa.schema([
    ("cricinfo_match_id", pa.int64()), ("scheduled_start_stated", pa.time32("s")), ("stated_time_zone", pa.string()),
    ("day_night_flag", pa.bool_()),
    ("time_text", pa.string()), ("date_text", pa.string()), ("template", pa.string()),
    ("page_title", pa.string()), ("page_revid", pa.int64()),
    ("duplicates", pa.int32()), ("duplicate_conflict", pa.bool_()),
])


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per match id: the first row (page order) with a time, else the first with a day/night flag, else
    the first. A conflict is two rows of one match with different times or flags."""
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["cricinfo_match_id"], []).append(r)
    out = []
    for mid in sorted(groups):
        g = groups[mid]
        keep = dict(min(g, key=lambda r: (r["scheduled_start_stated"] is None, r["day_night_flag"] is None, r["_order"])))
        times = {r["scheduled_start_stated"] for r in g} - {None}
        flags = {r["day_night_flag"] for r in g} - {None}
        keep.update(duplicates=len(g) - 1, duplicate_conflict=len(times) > 1 or len(flags) > 1)
        out.append(keep)
    return out


def build_wiki_series(snapshot_folder) -> None:
    snapshot_folder = Path(snapshot_folder)
    index = json.loads((snapshot_folder / INDEX_NAME).read_text(encoding="utf-8"))["pages"]
    rows = []
    for page_no, entry in enumerate(index):
        wikitext = bronze.read_zst(snapshot_folder / entry["stored_as"]).decode("utf-8")
        for row_no, row in enumerate(parse_page(entry["resolved_title"], wikitext)):
            row.update(page_revid=entry["revid"], _order=(page_no, row_no))
            rows.append(row)
    # pyarrow writes a time as UTC-adjusted, which DuckDB reads as TIME WITH TIME ZONE; the start is a local wall-clock
    # time, so it goes through as text and DuckDB writes a plain TIME
    out = [{k: r.get(k) for k in SCHEMA.names} for r in dedupe(rows)]
    for r in out:
        r["scheduled_start_stated"] = r["scheduled_start_stated"].isoformat() if r["scheduled_start_stated"] else None
    table = pa.Table.from_pylist(out, schema=SCHEMA.set(1, pa.field("scheduled_start_stated", pa.string())))
    con = duckdb.connect()
    con.register("schedule", table)
    con.execute(f"""COPY (SELECT * REPLACE (CAST(scheduled_start_stated AS TIME) AS scheduled_start_stated)
                    FROM schedule ORDER BY cricinfo_match_id)
                    TO '{SILVER / "wikipedia_match_schedule.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    con.close()
