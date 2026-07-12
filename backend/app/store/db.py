"""
store/db.py

SQLite fact store: filing metadata, canonicalized facts, and prose chunk
text.

Design principle
-----------------
This module holds no domain logic. It stores exactly what upstream
ingestion stages decide (canonicalized facts, filing metadata, prose
chunks) and answers deterministic lookups for downstream agents. It never
classifies a table, canonicalizes a metric label, or decides GAAP status -
those judgments happen in ingest/ before a row ever reaches this store.
Keeping the store dumb is what lets the Fact Retriever query it with plain
SQL and no LLM.

Schema
------
filings        one row per ingested filing: source metadata and citation URL.
facts          one row per (metric, period, filing): typed value with full
               provenance back to the exact page, table, and row.
prose_chunks   narrative text chunks with provenance. Embedded separately by
               store/vector_store.py; text is never embedded here.

The facts table is indexed on (company_ticker, metric_canonical_id, year,
quarter, is_gaap), matching the lookup pattern in
agents/fact_retriever.py and the production indexing plan from Part 8 of
the write-up.

Dependencies
------------
pydantic >= 2.0  (storage itself uses the stdlib sqlite3 module)

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "fact_store.db"


def _resolve_db_path() -> str:
    """Return the SQLite file path, honoring FACT_STORE_DB_PATH if set.

    Returns:
        Absolute path to the SQLite database file.
    """
    return os.environ.get("FACT_STORE_DB_PATH", str(DEFAULT_DB_PATH))


# =============================================================================
# Schema DDL
# =============================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    id              TEXT PRIMARY KEY,
    company_ticker  TEXT NOT NULL,
    form_type       TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    fiscal_quarter  INTEGER,
    period_end_date TEXT NOT NULL,
    filing_url      TEXT NOT NULL,
    filing_path     TEXT NOT NULL,
    is_amendment    INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker       TEXT NOT NULL,
    metric_canonical_id  TEXT NOT NULL,
    metric_raw_label     TEXT NOT NULL,
    value                REAL NOT NULL,
    units                TEXT NOT NULL,
    year                 INTEGER NOT NULL,
    quarter              INTEGER,
    is_ttm               INTEGER NOT NULL DEFAULT 0,
    is_gaap              INTEGER NOT NULL DEFAULT 1,
    is_restated          INTEGER NOT NULL DEFAULT 0,
    filing_id            TEXT NOT NULL REFERENCES filings(id),
    page_number          INTEGER NOT NULL,
    table_id             TEXT NOT NULL,
    row_id               INTEGER NOT NULL,
    ambiguity_flags      TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_lookup
    ON facts (company_ticker, metric_canonical_id, year, quarter, is_gaap);

-- One source row can carry multiple period columns (e.g. "Total revenues"
-- with a value for 2025, 2024, and 2023), so row_id alone is not a unique
-- key; year (and quarter) must be part of it. quarter is COALESCE'd to a
-- sentinel because SQLite treats every NULL as distinct in a UNIQUE index,
-- which would defeat de-duplication for full-year facts (quarter IS NULL).
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_provenance
    ON facts (filing_id, table_id, row_id, year, COALESCE(quarter, -1));

CREATE TABLE IF NOT EXISTS prose_chunks (
    id           TEXT PRIMARY KEY,
    filing_id    TEXT NOT NULL REFERENCES filings(id),
    section_type TEXT NOT NULL,
    heading      TEXT,
    page_start   INTEGER NOT NULL,
    page_end     INTEGER NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_prose_chunks_filing
    ON prose_chunks (filing_id);
"""


# =============================================================================
# Connection
# =============================================================================

def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection to the fact store, creating the schema if needed.

    Args:
        db_path: Override path to the SQLite file. Defaults to the
            FACT_STORE_DB_PATH environment variable, then DEFAULT_DB_PATH.

    Returns:
        A sqlite3.Connection with row_factory set to sqlite3.Row so callers
        can access columns by name, as agents/fact_retriever.py does.
    """
    path = db_path or _resolve_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


# =============================================================================
# Insert-parameter schemas
# =============================================================================

class FilingRecord(BaseModel):
    """Source metadata for one ingested filing."""
    id: str
    company_ticker: str
    form_type: str  # '10-K' | '10-Q' | '10-K/A' | '10-Q/A'
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    period_end_date: str  # ISO date, e.g. '2025-12-31'
    filing_url: str
    filing_path: str
    is_amendment: bool = False


class FactRecord(BaseModel):
    """A single canonicalized numeric fact with full provenance."""
    company_ticker: str
    metric_canonical_id: str
    metric_raw_label: str
    value: float
    units: str
    year: int
    quarter: Optional[int] = None
    is_ttm: bool = False
    is_gaap: bool = True
    is_restated: bool = False
    filing_id: str
    page_number: int
    table_id: str
    row_id: int
    ambiguity_flags: List[str] = Field(default_factory=list)


class ProseChunkRecord(BaseModel):
    """A narrative text chunk with provenance, prior to embedding."""
    id: str
    filing_id: str
    section_type: str
    heading: Optional[str] = None
    page_start: int
    page_end: int
    chunk_index: int
    text: str


# =============================================================================
# Writes
# =============================================================================

def insert_filing(conn: sqlite3.Connection, record: FilingRecord) -> None:
    """Insert or replace a filing's source metadata.

    Args:
        conn: Open connection from get_conn().
        record: Filing metadata to store.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO filings
            (id, company_ticker, form_type, fiscal_year, fiscal_quarter,
             period_end_date, filing_url, filing_path, is_amendment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id, record.company_ticker, record.form_type,
            record.fiscal_year, record.fiscal_quarter, record.period_end_date,
            record.filing_url, record.filing_path, int(record.is_amendment),
        ),
    )
    conn.commit()
    log.info(
        "Stored filing %s (%s %s)",
        record.id, record.company_ticker, record.form_type,
    )


def insert_fact(conn: sqlite3.Connection, record: FactRecord) -> int:
    """Insert or replace a single canonicalized fact.

    The provenance triple (filing_id, table_id, row_id) is the natural key,
    so re-ingesting the same filing overwrites the same rows instead of
    duplicating them.

    Args:
        conn: Open connection from get_conn().
        record: Fact to store.

    Returns:
        The row id of the inserted fact.
    """
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO facts
            (company_ticker, metric_canonical_id, metric_raw_label, value,
             units, year, quarter, is_ttm, is_gaap, is_restated, filing_id,
             page_number, table_id, row_id, ambiguity_flags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.company_ticker, record.metric_canonical_id,
            record.metric_raw_label, record.value, record.units, record.year,
            record.quarter, int(record.is_ttm), int(record.is_gaap),
            int(record.is_restated), record.filing_id, record.page_number,
            record.table_id, record.row_id,
            ",".join(record.ambiguity_flags) if record.ambiguity_flags else None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_prose_chunk(conn: sqlite3.Connection, record: ProseChunkRecord) -> None:
    """Insert or replace a prose chunk's text and provenance.

    Args:
        conn: Open connection from get_conn().
        record: Prose chunk to store.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO prose_chunks
            (id, filing_id, section_type, heading, page_start, page_end,
             chunk_index, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id, record.filing_id, record.section_type, record.heading,
            record.page_start, record.page_end, record.chunk_index,
            record.text,
        ),
    )
    conn.commit()


# =============================================================================
# CLI entry point (manual testing / one-off schema init)
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Initialize (or verify) the SQLite fact store schema.",
    )
    ap.add_argument("--db-path", default=None, help="Override DB file path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = get_conn(args.db_path)
    tables = [
        r[0] for r in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    log.info(
        "Fact store ready at %s. Tables: %s",
        args.db_path or _resolve_db_path(), tables,
    )
    conn.close()
