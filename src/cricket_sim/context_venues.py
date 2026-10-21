"""Compatibility shim: the venue context now lives in the cricket_sim.venues package.

Public entry points (unchanged): ingest_venues_and_weather() and build_venue_context(con).
"""

from .venues.build import build_venue_context  # noqa: F401
from .venues.canonical import canonicalize, core_name, load_overrides, norm, sig_tokens, slugify  # noqa: F401
from .venues.ingest import ingest_venues_and_weather  # noqa: F401

__all__ = ["build_venue_context", "ingest_venues_and_weather", "canonicalize", "load_overrides", "norm", "slugify",
           "core_name", "sig_tokens"]
