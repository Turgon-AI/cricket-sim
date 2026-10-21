"""Cricsheet's own list of matches it knows it is missing (https://cricsheet.org/missing/)."""

import re
from datetime import date
from html import unescape

import pyarrow as pa
import pyarrow.parquet as pq

from .bronze import read_zst
from .paths import SILVER

SCHEMA = pa.schema([
    ("list_type", pa.string()), ("list_name", pa.string()), ("gender", pa.string()),
    ("start_date", pa.date32()), ("teams", pa.string()), ("entry_number", pa.int32()),
])

TOKEN = re.compile(r"<h4>(.*?)</h4>|<h5>(.*?)</h5>|<h6>(.*?)</h6>|<dt>(.*?)</dt>|<dd>(.*?)</dd>", re.S)


def parse_missing_html(html: str) -> list[dict]:
    rows, list_type, list_name, gender, day = [], None, None, None, None
    for h4, h5, h6, dt, dd in TOKEN.findall(html):
        if h4:
            list_type = {"By match type": "match_type", "By competition": "competition"}.get(unescape(h4).strip())
        elif h5:
            list_name = unescape(h5).strip().removesuffix(" Matches")
        elif h6:
            gender = unescape(h6).strip().split()[0].lower()  # "Male matches" -> male
        elif dt:
            day = date.fromisoformat(dt.strip())
        elif dd and list_type and day:
            rows.append({"list_type": list_type, "list_name": list_name, "gender": gender,
                         "start_date": day, "teams": unescape(dd).strip()})
    for n, row in enumerate(rows, start=1):
        row["entry_number"] = n
    return rows


def build_cricsheet_missing(snapshot_folder) -> None:
    raw = read_zst(snapshot_folder / "missing.html.zst")
    rows = parse_missing_html(raw.decode("utf-8"))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), SILVER / "cricsheet_missing.parquet")
