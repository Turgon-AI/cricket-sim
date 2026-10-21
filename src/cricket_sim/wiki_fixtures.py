"""Reference list of numbered men's internationals from Wikipedia season pages.

Bronze source `wiki_fixtures`: wikitext of every "International cricket in YYYY" / "YYYY–YY" page,
pinned by revision id. Silver: data/silver/wikipedia_fixtures.parquet, one row per (match_format, match_number).
Licence CC BY-SA 4.0 (see config/wiki_sources.yml).
"""

import json
import re
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml

from . import bronze
from .paths import SILVER, load_yaml

SOURCE = "wiki_fixtures"
INDEX_NAME = "index.json"
MAX_ATTEMPTS = 6


# --------------------------------------------------------------------------------------------- ingest

def page_titles(cfg: dict) -> tuple[list[str], list[str]]:
    """(required titles, optional titles) in season order."""
    titles = [f"International cricket in {cfg['first_season']}"]
    for year in range(cfg["first_year"], cfg["last_year"] + 1):
        titles.append(f"International cricket in {year}")
        titles.append(f"International cricket in {year}–{(year + 1) % 100:02d}")
    return titles, list(cfg.get("optional_pages", []))


def page_slug(title: str) -> str:
    return title.removeprefix("International cricket in ").replace("–", "-").replace(" ", "_")


def _fetch_page(session: requests.Session, endpoint: str, title: str) -> dict | None:
    params = {"action": "query", "prop": "revisions", "rvprop": "content|ids", "rvslots": "main",
              "titles": title, "format": "json", "formatversion": "2", "redirects": "1"}
    for attempt in range(MAX_ATTEMPTS):
        r = session.get(endpoint, params=params, timeout=60)
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS - 1:
            break
        wait = float(r.headers.get("Retry-After") or 0) or 5.0 * 2 ** attempt
        print(f"  wiki_fixtures: HTTP {r.status_code} on {title!r}; retrying in {wait:.0f}s")
        time.sleep(min(wait, 120.0))
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"wiki_fixtures: API error for {title!r}: {data['error']}")
    page = data["query"]["pages"][0]
    if page.get("missing") or page.get("invalid"):
        return None
    rev = page["revisions"][0]
    return {"title": title, "resolved_title": page["title"], "pageid": page["pageid"],
            "revid": rev["revid"], "parentid": rev.get("parentid"), "timestamp": rev.get("timestamp"),
            "wikitext": rev["slots"]["main"]["content"]}


def ingest_wiki_fixtures() -> str:
    from .ingest import _finish, _start

    cfg = load_yaml("wiki_sources.yml")[SOURCE]
    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    required, optional = page_titles(cfg)
    folder = _start(SOURCE, snapshot_id)
    if folder is None:
        return snapshot_id
    session = requests.Session()
    session.headers["User-Agent"] = cfg["user_agent"]
    fetched = []
    try:
        for title in required + optional:
            page = _fetch_page(session, cfg["endpoint"], title)
            time.sleep(cfg.get("delay_seconds", 1.0))
            if page is None:
                if title in optional:
                    print(f"  wiki_fixtures: optional page {title!r} does not exist; skipped")
                    continue
                raise RuntimeError(f"wiki_fixtures: required page {title!r} is missing")
            fetched.append(page)
            print(f"  wiki_fixtures: {title} rev {page['revid']} ({len(page['wikitext'])} chars)")
    except BaseException:
        bronze.discard_partial(folder)  # never leave a half-written snapshot behind
        raise

    files, index = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for page in fetched:
            name = f"{page_slug(page['title'])}.wiki"
            raw = Path(tmp) / name
            raw.write_text(page["wikitext"], encoding="utf-8")
            raw_sha = bronze.compress_to_zst(raw, folder / f"{name}.zst")
            files.append({"name": name, "title": page["title"], "revid": page["revid"],
                          "url": f"https://en.wikipedia.org/w/index.php?oldid={page['revid']}",
                          "stored_as": f"{name}.zst", "compressed": True,
                          "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha})
            index.append({k: page[k] for k in ("title", "resolved_title", "pageid", "revid", "parentid", "timestamp")}
                         | {"stored_as": f"{name}.zst"})
    (folder / INDEX_NAME).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files.append({"name": INDEX_NAME, "stored_as": INDEX_NAME, "compressed": False})
    upstream = {"endpoint": cfg["endpoint"], "pages": len(fetched), "licence": cfg["licence"]}
    return _finish(SOURCE, snapshot_id, upstream, files)


# --------------------------------------------------------------------------------------------- parse

# Cricinfo-style codes and spelled-out names used in {{cr|...}} templates and headings -> Cricsheet team names.
TEAM_CODES = {
    "AFG": "Afghanistan", "ARG": "Argentina", "AUS": "Australia", "AUT": "Austria", "BAH": "Bahamas",
    "BAN": "Bangladesh", "BGD": "Bangladesh", "BAR": "Barbados", "BEL": "Belgium", "BER": "Bermuda",
    "BHR": "Bahrain", "BHU": "Bhutan", "BIZ": "Belize", "BOT": "Botswana", "BWA": "Botswana", "BRA": "Brazil",
    "BRU": "Brunei", "BUL": "Bulgaria", "CAM": "Cameroon", "CAN": "Canada", "CAY": "Cayman Islands",
    "CYM": "Cayman Islands", "CHN": "China", "CHI": "Chile", "COK": "Cook Islands", "CRC": "Costa Rica",
    "CRO": "Croatia", "CYP": "Cyprus", "CZE": "Czech Republic", "DEN": "Denmark", "ENG": "England",
    "ESP": "Spain", "SPA": "Spain", "EST": "Estonia", "FIJ": "Fiji", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAM": "Gambia", "GER": "Germany", "GHA": "Ghana", "GIB": "Gibraltar", "GRE": "Greece",
    "GUE": "Guernsey", "GGY": "Guernsey", "HK": "Hong Kong", "HKG": "Hong Kong", "HUN": "Hungary",
    "IDN": "Indonesia", "INA": "Indonesia", "IND": "India", "IOM": "Isle of Man", "IRE": "Ireland",
    "IRN": "Iran", "ISR": "Israel", "ITA": "Italy", "CIV": "Ivory Coast", "JPN": "Japan", "JER": "Jersey",
    "JEY": "Jersey", "KEN": "Kenya", "KOR": "South Korea", "KSA": "Saudi Arabia", "SAU": "Saudi Arabia",
    "KUW": "Kuwait", "LES": "Lesotho", "LUX": "Luxembourg", "MAS": "Malaysia", "MYS": "Malaysia",
    "MDV": "Maldives", "MLI": "Mali", "MLT": "Malta", "MEX": "Mexico", "MGL": "Mongolia", "MNG": "Mongolia",
    "MOZ": "Mozambique", "MWI": "Malawi", "MYA": "Myanmar", "NAM": "Namibia", "NED": "Netherlands",
    "NLD": "Netherlands", "NEP": "Nepal", "NPL": "Nepal", "NGR": "Nigeria", "NGA": "Nigeria", "NOR": "Norway",
    "NZ": "New Zealand", "NZL": "New Zealand", "OMA": "Oman", "OMN": "Oman", "PAK": "Pakistan",
    "PAN": "Panama", "PER": "Peru", "PHI": "Philippines", "PNG": "Papua New Guinea", "POR": "Portugal",
    "QAT": "Qatar", "ROU": "Romania", "RSA": "South Africa", "SA": "South Africa", "SAF": "South Africa",
    "ZAF": "South Africa", "RWA": "Rwanda", "SAM": "Samoa", "SCO": "Scotland", "SRB": "Serbia",
    "SEY": "Seychelles", "SIN": "Singapore", "SGP": "Singapore", "SLE": "Sierra Leone", "SL": "Sri Lanka",
    "SRI": "Sri Lanka", "LKA": "Sri Lanka", "SLO": "Slovenia", "SUR": "Suriname", "SWE": "Sweden",
    "SUI": "Switzerland", "SWZ": "Eswatini", "TAN": "Tanzania", "TZA": "Tanzania", "THA": "Thailand",
    "TLS": "Timor-Leste", "TTO": "Trinidad and Tobago", "TUR": "Turkey", "TCA": "Turks and Caicos Islands",
    "UAE": "United Arab Emirates", "UGA": "Uganda", "USA": "United States of America", "UZB": "Uzbekistan",
    "VAN": "Vanuatu", "WI": "West Indies", "WIN": "West Indies", "ZAM": "Zambia", "ZMB": "Zambia",
    "ZIM": "Zimbabwe",
}
TEAM_NAMES = {name.lower(): name for name in TEAM_CODES.values()} | {
    "united states": "United States of America", "us": "United States of America",
    "the united states": "United States of America", "uae": "United Arab Emirates",
    "the west indies": "West Indies", "the netherlands": "Netherlands", "holland": "Netherlands",
    "the uae": "United Arab Emirates", "png": "Papua New Guinea", "the caribbean": "West Indies",
    "turks and caicos islands": "Turks and Caicos Islands", "the united arab emirates": "United Arab Emirates",
    "new zealanders": "New Zealand", "australians": "Australia", "indians": "India", "pakistanis": "Pakistan", "swaziland": "Eswatini",
    "icc world xi": "ICC World XI", "world xi": "ICC World XI", "asia xi": "Asia XI", "africa xi": "Africa XI",
}

NUM_RE = re.compile(r"(?<![A-Za-z])(Test|ODI|T20I)\s+(\d{1,4})\b")
CR_RE = re.compile(r"\{\{\s*(cr|crw|cricon|cr-rt)\s*\|\s*([^|}]+?)\s*(?:\|[^}]*)?\}\}", re.I)
URL_RE = re.compile(r"https?://[^\s\]|]+")
HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())} | {"sept": 9})
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_DM_RE = re.compile(rf"^\D*?(\d{{1,2}})(?:\s*(?:–|-|—|−|&|,|and)\s*\d{{1,2}})*\s+({MONTH_ALT})\.?\b(?:,?\s+(\d{{4}}))?", re.I)
DATE_MD_RE = re.compile(rf"^\W*({MONTH_ALT})\.?\s+(\d{{1,2}})\b(?:,?\s+(\d{{4}}))?", re.I)
PAGE_YEAR_RE = re.compile(r"(\d{4})(?:[–-](\d{2}))?\s*$")


def _split_top(text: str, sep: str) -> list[str]:
    """Split on sep only outside [[...]] and {{...}}."""
    parts, depth_link, depth_tmpl, start, i = [], 0, 0, 0, 0
    while i < len(text):
        two = text[i:i + 2]
        if two == "[[":
            depth_link += 1; i += 2; continue
        if two == "]]" and depth_link:
            depth_link -= 1; i += 2; continue
        if two == "{{":
            depth_tmpl += 1; i += 2; continue
        if two == "}}" and depth_tmpl:
            depth_tmpl -= 1; i += 2; continue
        if not depth_link and not depth_tmpl and text.startswith(sep, i):
            parts.append(text[start:i]); i += len(sep); start = i; continue
        i += 1
    parts.append(text[start:])
    return parts


def _strip_attrs(cell: str) -> str:
    """'style="x" | content' -> 'content' (only when the prefix looks like HTML attributes)."""
    pieces = _split_top(cell, "|")
    if len(pieces) > 1 and ("=" in pieces[0] or pieces[0].strip() in ("", ";")) and "[" not in pieces[0]:
        return "|".join(pieces[1:])
    return cell


def team_from_token(token: str | None) -> str | None:
    if not token:
        return None
    token = token.strip()
    return TEAM_CODES.get(token.upper()) or TEAM_NAMES.get(token.lower())


def clean_text(text: str) -> str:
    """Wikitext -> plain text: refs/comments dropped, links and team templates rendered."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = CR_RE.sub(lambda m: team_from_token(m.group(2)) or m.group(2), text)
    text = re.sub(r"\{\{\s*(?:dts|date|nowrap|small)\s*\|([^{}]*)\}\}", lambda m: m.group(1).split("|")[0], text, flags=re.I)
    text = re.sub(r"\{\{\s*(?:DLS|D/L|DLMethod)\s*\}\}", "(DLS)", text, flags=re.I)
    for _ in range(3):  # drop any other templates (webarchive, refs, flags), innermost first
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def cricinfo_id(cell: str) -> int | None:
    """Cricinfo match id from a scorecard URL (old /match/N.html, /scorecard/N/, or new slug-N/full-scorecard)."""
    for url in URL_RE.findall(cell):
        idx = url.find("cricinfo.com")
        if idx < 0:
            continue
        path = url[idx:]
        for pat in (r"/match/(\d+)", r"/scorecard/(\d+)", r"/game/(\d+)",
                    r"/series/[^/]+/[a-z0-9-]*?-(\d+)/(?:full-scorecard|live-cricket-score|match-report|"
                    r"ball-by-ball-commentary|scorecard|match-preview)",
                    r"/series/[^/]+/[a-z0-9-]*?-vs?-[a-z0-9-]*?-(\d+)/?"):
            m = re.search(pat, path)
            if m:
                return int(m.group(1))
    return None


def page_years(title: str) -> tuple[int, int]:
    m = PAGE_YEAR_RE.search(title)
    if not m:
        raise ValueError(f"cannot read a season from {title!r}")
    first = int(m.group(1))
    return (first, first + 1) if m.group(2) else (first, first)


def resolve_start_date(date_text: str, title: str) -> date | None:
    """First day of the date cell. Split-season pages: Jul–Dec -> first year, Jan–Jun -> second year."""
    text = date_text.strip()
    m = DATE_DM_RE.search(text)
    if m:
        day, month, year = int(m.group(1)), MONTHS[m.group(2).lower()], m.group(3)
    else:
        m = DATE_MD_RE.search(text)
        if not m:
            return None
        month, day, year = MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3)
    if year is None:
        first, second = page_years(title)
        year = first if month >= 7 or first == second else second
    try:
        return date(int(year), month, day)
    except ValueError:
        return None


def classify_result(result: str) -> tuple[str, bool]:
    """(result_class, is_abandoned_no_ball)."""
    r = result.lower()
    if not r:
        return "unknown", False
    no_play = bool(re.search(r"without a ball|no ball (?:was )?bowled|no play|without play|not a ball", r))
    if "abandon" in r or "cancel" in r or no_play:
        played = bool(re.search(r"\d+\s*/\s*\d+|\d+ runs|\d+ wickets|after \d|overs", r)) and not no_play
        return "abandoned", not played
    if "tie" in r:  # "Match tied", "tied (X won super over)"
        return "tie", False
    if "drawn" in r or re.search(r"\bdraw\b", r):
        return "draw", False
    if "no result" in r:
        return "no_result", False
    if re.search(r"\bwon\b|\bby\b.*\b(runs?|wickets?|wkts?|innings|inns)\b|\bconceded\b|\bforfeit|\bawarded\b", r):
        return "result", False
    return "unknown", False


def heading_teams(heading: str | None) -> tuple[str | None, str | None, str]:
    """Teams implied by a section heading: 'A in B' -> (B, A) home first; 'A v B' -> (A, B)."""
    if not heading:
        return None, None, "none"
    h = clean_text(heading)
    h = re.sub(r"^\d{4}(?:[–-]\d{2,4})?\s+", "", h)
    h = re.sub(r"\s*\([^)]*\)\s*$", "", h)  # "(April 2023)", "(Test leg)"
    m = re.match(r"^(.+?)\s+(?:v|vs\.?|versus|against)\s+(.+?)(?:\s+in\s+.+)?$", h, re.I)
    if m:
        a, b = team_from_token(m.group(1)), team_from_token(m.group(2))
        if a and b:
            return a, b, "heading"
    m = re.match(r"^(.+?)\s+in\s+(.+)$", h, re.I)
    if m:
        away = team_from_token(m.group(1))
        hosts = [team_from_token(x) for x in re.split(r"\s*(?:,|\band\b|&)\s*", m.group(2)) if x.strip()]
        if away and hosts and hosts[0]:
            return hosts[0], away, ("heading" if len(hosts) == 1 else "heading_multi_host")
    return None, None, "none"


def _iter_rows(wikitext: str):
    """Yield (section_heading, header_cells, row_cells) for every data row of every wikitable."""
    heading, header, row, in_header_row = None, [], None, False
    headings, pending_header = {}, []

    def flush():
        nonlocal row
        if row:
            yield heading, list(header), row
        row = None

    for line in wikitext.split("\n"):
        stripped = line.strip()
        hm = HEADING_RE.match(stripped)
        if hm:
            yield from flush()
            level = len(hm.group(1))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = hm.group(2)
            heading = headings[max(headings)]
            header = []
            continue
        if stripped.startswith("{|") or stripped.startswith("|}") or stripped.startswith("|-"):
            yield from flush()
            if stripped.startswith("{|"):
                header = []
            in_header_row = False
            continue
        if stripped.startswith("!"):
            yield from flush()
            cells = [clean_text(_strip_attrs(c)) for c in _split_top(stripped[1:], "!!")]
            if not in_header_row:
                pending_header = []
            pending_header += cells
            in_header_row = True
            if len(pending_header) >= 3:
                header = pending_header
            continue
        if stripped.startswith("|") and not stripped.startswith("|+"):
            cells = [_strip_attrs(c).strip() for c in _split_top(stripped[1:], "||")]
            if row is None:
                row = []
            row.extend(cells)
            continue
        if row is not None and stripped:
            row[-1] += " " + stripped  # continuation of a multi-line cell
    yield from flush()


def parse_wikitext(title: str, wikitext: str) -> list[dict]:
    """Every numbered men's Test / ODI / T20I row on one season page."""
    out = []
    for heading, header, cells in _iter_rows(wikitext):
        num_idx = next((i for i, c in enumerate(cells[:3]) if NUM_RE.search(clean_text(c))), None)
        if num_idx is None:
            continue
        fmt, number = NUM_RE.search(clean_text(cells[num_idx])).groups()
        raw_row = " ".join(cells)
        is_women = bool(re.search(r"\{\{\s*crw\s*\|", raw_row, re.I)) or bool(
            heading and re.search(r"\bwomen", heading, re.I))
        if is_women:
            continue
        rest = cells[num_idx + 1:]
        date_idx = next((i for i, c in enumerate(rest[:2]) if resolve_start_date(clean_text(c), title)), 0)
        date_text = clean_text(rest[date_idx]) if rest else ""
        result_raw = rest[-1] if len(rest) >= 2 else ""
        middle = rest[date_idx + 1:-1]
        team_cells = [team_from_token(m.group(2)) for c in middle for m in [CR_RE.search(c)] if m]
        team_cells = [t for t in team_cells if t]
        team1 = team2 = None
        teams_source = "none"
        if len(team_cells) >= 2:
            team1, team2, teams_source = team_cells[0], team_cells[1], "row"
        else:
            cap = [team_from_token(re.sub(r"\s+captain$", "", h, flags=re.I)) for h in header
                   if re.search(r"\scaptain$", h, re.I) and not re.match(r"^(home|away|team|captain)", h, re.I)]
            cap = [t for t in cap if t]
            if len(cap) >= 2:
                team1, team2, teams_source = cap[0], cap[1], "header"
            else:
                team1, team2, teams_source = heading_teams(heading)
        result_text = clean_text(result_raw)
        result_class, abandoned = classify_result(result_text)
        teams = {team1, team2}
        lead = CR_RE.match(result_raw.strip())
        winner = team_from_token(lead.group(2)) if lead and result_class == "result" else None
        out.append({
            "match_format": fmt, "match_number": int(number), "cricinfo_match_id": cricinfo_id(cells[num_idx]),
            "date_text": date_text, "start_date": resolve_start_date(date_text, title),
            "team1": team1, "team2": team2, "teams_source": teams_source,
            "result_text": result_text, "result_class": result_class, "is_abandoned_no_ball": abandoned,
            "involves_afghanistan": "Afghanistan" in teams or "afghanistan" in clean_text(raw_row).lower(),
            "page_title": title, "section_heading": clean_text(heading) if heading else None,
            "_winner": winner,
        })
    _correct_heading_teams(out)
    for row in out:
        winner = row.pop("_winner")
        flags = []
        if row["team1"] and row["team1"] == row["team2"]:
            flags.append("same_team")
        if winner and row["team1"] and row["team2"] and winner not in (row["team1"], row["team2"]):
            flags.append("winner_not_in_teams")
        row["qa_flags"] = ",".join(flags) or None
        row["involves_afghanistan"] = row["involves_afghanistan"] or "Afghanistan" in (row["team1"], row["team2"])
    return out


def _correct_heading_teams(rows: list[dict]) -> None:
    """'Australia in the United Arab Emirates' really meant Pakistan v Australia at a neutral venue: when the
    winners in a heading-derived section include exactly one team outside the heading's pair, it replaces the
    host for the whole section; with more than one outsider the heading teams are dropped."""
    sections: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["teams_source"] in ("heading", "heading_multi_host"):
            sections.setdefault((row["section_heading"], row["team1"], row["team2"]), []).append(row)
    for (_, home, away), group in sections.items():
        outsiders = {r["_winner"] for r in group if r["_winner"] and r["_winner"] not in (home, away)}
        if len(outsiders) == 1:
            new_home = outsiders.pop()
            for r in group:
                r.update(team1=new_home, teams_source="heading_corrected")
        elif outsiders:
            for r in group:
                r.update(team1=None, team2=None, teams_source="heading_unreliable")


# --------------------------------------------------------------------------------------------- build

SCHEMA = pa.schema([
    ("match_format", pa.string()), ("match_number", pa.int32()), ("cricinfo_match_id", pa.int64()),
    ("date_text", pa.string()), ("start_date", pa.date32()),
    ("team1", pa.string()), ("team2", pa.string()), ("teams_source", pa.string()),
    ("result_text", pa.string()), ("result_class", pa.string()), ("is_abandoned_no_ball", pa.bool_()),
    ("involves_afghanistan", pa.bool_()), ("page_title", pa.string()), ("page_revid", pa.int64()),
    ("section_heading", pa.string()),
    ("qa_flags", pa.string()),
    ("duplicates", pa.int32()), ("duplicate_conflict", pa.bool_()), ("conflict_fields", pa.string()),
])


def _completeness(row: dict) -> int:
    return sum(row[k] is not None for k in ("cricinfo_match_id", "start_date", "team1", "team2")) \
        + (row["result_class"] != "unknown")


def _conflicts(rows: list[dict]) -> list[str]:
    fields = []
    for name, key in (("cricinfo_match_id", lambda r: r["cricinfo_match_id"]),
                      ("start_date", lambda r: r["start_date"]),
                      ("teams", lambda r: frozenset((r["team1"], r["team2"])) if r["team1"] and r["team2"] else None)):
        values = {key(r) for r in rows} - {None}
        if len(values) > 1:
            fields.append(name)
    return fields


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per (match_format, match_number). Rows must carry _order (page order, row order)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["match_format"], row["match_number"]), []).append(row)
    # Dates of unambiguous numbers, used to pick the right row when a number was given to two different matches.
    clean_dates = {key: g[0]["start_date"] for key, g in groups.items()
                   if not _conflicts(g) and any(r["start_date"] for r in g)}
    clean_dates = {key: next(r["start_date"] for r in groups[key] if r["start_date"]) for key in clean_dates}

    def fits_sequence(row: dict) -> int:
        fmt, num, day = row["match_format"], row["match_number"], row["start_date"]
        if day is None:
            return 1
        lower = [d for (f, n), d in clean_dates.items() if f == fmt and num - 5 <= n < num]
        upper = [d for (f, n), d in clean_dates.items() if f == fmt and num < n <= num + 5]
        return 0 if (not lower or max(lower) <= day) and (not upper or day <= min(upper)) else 1

    out = []
    for key in sorted(groups):
        group = groups[key]
        conflicts = _conflicts(group)
        rank = (lambda r: (fits_sequence(r), -_completeness(r), r["_order"])) if conflicts \
            else (lambda r: (-_completeness(r), r["_order"]))
        keep = dict(min(group, key=rank))
        keep.update(duplicates=len(group) - 1, duplicate_conflict=bool(conflicts),
                    conflict_fields=",".join(conflicts) or None)
        out.append(keep)
    return out


def build_wiki_fixtures(snapshot_folder) -> None:
    snapshot_folder = Path(snapshot_folder)
    index = json.loads((snapshot_folder / INDEX_NAME).read_text(encoding="utf-8"))
    rows = []
    for page_no, entry in enumerate(index):
        wikitext = bronze.read_zst(snapshot_folder / entry["stored_as"]).decode("utf-8")
        for row_no, row in enumerate(parse_wikitext(entry["title"], wikitext)):
            row.update(page_revid=entry["revid"], _order=(page_no, row_no))
            rows.append(row)
    table = pa.Table.from_pylist([{k: r.get(k) for k in SCHEMA.names} for r in dedupe(rows)], schema=SCHEMA)
    pq.write_table(table, SILVER / "wikipedia_fixtures.parquet", compression="zstd")


# --------------------------------------------------------------------------------------------- linking to Cricsheet

# The strict rule: a link needs the two teams (unordered) to agree and start dates within one day; the
# id or the number only ranks the candidates. Reviewed exceptions are in config/link_overrides.yml, never in code.
LINK_METHODS = {"cricinfo_id": 3, "number_and_teams": 2, "date_and_teams": 1}
OVERRIDE_METHOD = "reviewed_override"
LINK_MAX_DAYS = 1
OVERRIDE_FIELDS = ("segment", "reference_number", "reference_cricinfo_id", "cricsheet_match_id", "cricsheet_number", "reason")


class LinkOverrideError(ValueError):
    """config/link_overrides.yml cannot be applied as written. Never skipped: the build stops."""


def load_link_overrides(path: Path | None = None) -> list[dict]:
    from .paths import CONFIG
    data = yaml.safe_load((path or CONFIG / "link_overrides.yml").read_text()) or {}
    return list(data.get("links") or [])


def _link_method(ref: dict, cs: dict) -> str | None:
    """The method this (reference row, Cricsheet match) pair supports under the strict rule, or None."""
    if ref["team1"] is None or abs((cs["start_date"] - ref["start_date"]).days) > LINK_MAX_DAYS:
        return None
    if sorted([ref["team1"], ref["team2"] or ""]) != sorted([cs["team1"], cs["team2"]]):
        return None
    if ref["cricinfo_match_id"] is not None and str(ref["cricinfo_match_id"]) == cs["match_id"]:
        return "cricinfo_id"
    if ref["match_number"] == cs["match_type_number"]:
        return "number_and_teams"
    return "date_and_teams"


def link_reference(refs: list[dict], cricsheet: list[dict], overrides=(), check_stale: bool = True) -> list[dict]:
    """One-to-one links between reference rows and Cricsheet matches of the same format.

    Strict pass: every pair the strict rule supports is scored (method strength, then fewer days apart) and linked
    greedily from the best pair down, each side used at most once, so a wrong id cannot take a match from its correct
    pair and a runner-up is tried when the best candidate is taken. Then the reviewed overrides are added
    (link_method 'reviewed_override'). An override must name rows that exist with the identifiers it gives, must have a
    reason, and must not touch a row the strict pass already linked: anything else raises LinkOverrideError.
    check_stale=False (a fixture SUBSET of the real snapshots) skips entries whose rows are not in the build."""
    by_day = {}
    for c in cricsheet:
        by_day.setdefault((c["segment"], c["start_date"].toordinal()), []).append(c)
    pairs = []
    for r in refs:
        day = r["start_date"].toordinal()
        for c in [c for d in range(day - LINK_MAX_DAYS, day + LINK_MAX_DAYS + 1) for c in by_day.get((r["segment"], d), [])]:
            method = _link_method(r, c)
            if method:
                pairs.append((-LINK_METHODS[method], abs((c["start_date"] - r["start_date"]).days),
                              r["segment"], r["match_number"], c["match_id"], method))
    links, ref_link, cs_link = [], {}, {}
    for _, _, segment, number, match_id, method in sorted(pairs):
        if (segment, number) in ref_link or match_id in cs_link:
            continue
        ref_link[(segment, number)] = match_id
        cs_link[match_id] = (segment, number)
        links.append({"segment": segment, "match_number": number, "cricsheet_match_id": match_id, "link_method": method})

    ref_by_key = {(r["segment"], r["match_number"]): r for r in refs}
    cs_by_id = {c["match_id"]: c for c in cricsheet}
    errors, seen = [], set()
    for o in overrides:
        missing = [f for f in OVERRIDE_FIELDS if f not in o]
        key, match_id = (o.get("segment"), o.get("reference_number")), str(o.get("cricsheet_match_id"))
        label = f"{key[0]} {key[1]} -> {match_id}"
        r, c = ref_by_key.get(key), cs_by_id.get(match_id)
        if missing:
            errors.append(f"{label}: missing {missing}")
        elif not str(o["reason"] or "").strip():
            errors.append(f"{label}: needs a reason")
        elif key in seen or match_id in {m for _, m in seen}:
            errors.append(f"{label}: listed twice")
        elif (r is None or c is None) and not check_stale:
            continue
        elif r is None or c is None or c["segment"] != key[0]:
            errors.append(f"{label}: no such reference row or Cricsheet match in this build")
        elif r["cricinfo_match_id"] != o["reference_cricinfo_id"] or c["match_type_number"] != o["cricsheet_number"]:
            errors.append(f"{label}: identifiers do not match the data (reference id {r['cricinfo_match_id']}, "
                          f"Cricsheet number {c['match_type_number']})")
        elif ref_link.get(key) == match_id:
            errors.append(f"{label}: not needed, the strict rule already makes this link; remove the entry")
        elif key in ref_link:
            errors.append(f"{label}: contradicts the strict link to {ref_link[key]}")
        elif match_id in cs_link:
            errors.append(f"{label}: Cricsheet match already linked to {cs_link[match_id][0]} {cs_link[match_id][1]}")
        else:
            seen.add((key, match_id))
            ref_link[key], cs_link[match_id] = match_id, key
            links.append({"segment": key[0], "match_number": key[1], "cricsheet_match_id": match_id,
                          "link_method": OVERRIDE_METHOD})
    if errors:
        raise LinkOverrideError("config/link_overrides.yml cannot be applied:\n  " + "\n  ".join(errors))
    return links
