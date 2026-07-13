"""Tests for store/db.py: schema, inserts, and the provenance unique index."""

from __future__ import annotations

from app.store.db import FactRecord, FilingRecord, ProseChunkRecord, insert_fact, insert_filing, insert_prose_chunk


def _filing(conn, filing_id="TSLA-10K-2025-12-31"):
    insert_filing(conn, FilingRecord(
        id=filing_id, company_ticker="TSLA", form_type="10-K",
        fiscal_year=2025, period_end_date="2025-12-31",
        filing_url="file:///x.pdf", filing_path="/x.pdf",
    ))
    return filing_id


def test_schema_creates_all_tables(db_conn):
    tables = {r[0] for r in db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"filings", "facts", "prose_chunks"} <= tables


def test_insert_and_query_fact_round_trip(db_conn):
    filing_id = _filing(db_conn)
    insert_fact(db_conn, FactRecord(
        company_ticker="TSLA", metric_canonical_id="METRIC_TOTAL_REVENUE",
        metric_raw_label="Total revenues", value=94827.0, units="USD_millions",
        year=2025, filing_id=filing_id, page_number=61,
        table_id=f"{filing_id}::page61::table1", row_id=7,
    ))
    row = db_conn.execute(
        "SELECT value, units FROM facts WHERE metric_canonical_id='METRIC_TOTAL_REVENUE'"
    ).fetchone()
    assert row["value"] == 94827.0
    assert row["units"] == "USD_millions"


def test_multi_period_same_row_id_are_distinct_facts(db_conn):
    """A single source row (e.g. 'Total revenues') carries a value per period
    column; row_id alone must not collide them under the unique index."""
    filing_id = _filing(db_conn)
    for year, value in [(2025, 94827.0), (2024, 97690.0), (2023, 96773.0)]:
        insert_fact(db_conn, FactRecord(
            company_ticker="TSLA", metric_canonical_id="METRIC_TOTAL_REVENUE",
            metric_raw_label="Total revenues", value=value, units="USD_millions",
            year=year, filing_id=filing_id, page_number=61,
            table_id=f"{filing_id}::page61::table1", row_id=7,
        ))
    rows = db_conn.execute("SELECT year, value FROM facts WHERE row_id=7").fetchall()
    assert len(rows) == 3


def test_reingesting_same_fact_is_idempotent(db_conn):
    filing_id = _filing(db_conn)
    record = FactRecord(
        company_ticker="TSLA", metric_canonical_id="METRIC_TOTAL_REVENUE",
        metric_raw_label="Total revenues", value=94827.0, units="USD_millions",
        year=2025, filing_id=filing_id, page_number=61,
        table_id=f"{filing_id}::page61::table1", row_id=7,
    )
    insert_fact(db_conn, record)
    insert_fact(db_conn, record)
    rows = db_conn.execute("SELECT * FROM facts WHERE row_id=7").fetchall()
    assert len(rows) == 1


def test_discrete_quarter_and_ytd_same_quarter_are_distinct(db_conn):
    """A 10-Q row can show both a discrete-quarter figure and a
    year-to-date-through-that-quarter figure under the same quarter number;
    is_ytd must be part of the uniqueness key or one silently overwrites
    the other."""
    filing_id = _filing(db_conn)
    insert_fact(db_conn, FactRecord(
        company_ticker="AAPL", metric_canonical_id="METRIC_TOTAL_REVENUE",
        metric_raw_label="Total net sales", value=95359.0, units="USD_millions",
        year=2025, quarter=2, is_ytd=False, filing_id=filing_id, page_number=4,
        table_id=f"{filing_id}::page4::table1", row_id=1,
    ))
    insert_fact(db_conn, FactRecord(
        company_ticker="AAPL", metric_canonical_id="METRIC_TOTAL_REVENUE",
        metric_raw_label="Total net sales", value=219659.0, units="USD_millions",
        year=2025, quarter=2, is_ytd=True, filing_id=filing_id, page_number=4,
        table_id=f"{filing_id}::page4::table1", row_id=1,
    ))
    rows = db_conn.execute(
        "SELECT value, is_ytd FROM facts WHERE row_id=1 ORDER BY is_ytd"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["value"] == 95359.0 and rows[0]["is_ytd"] == 0
    assert rows[1]["value"] == 219659.0 and rows[1]["is_ytd"] == 1


def test_insert_prose_chunk_round_trip(db_conn):
    filing_id = _filing(db_conn)
    insert_prose_chunk(db_conn, ProseChunkRecord(
        id=f"{filing_id}::risk::0", filing_id=filing_id, section_type="risk_factors",
        heading="Item 1A", page_start=18, page_end=54, chunk_index=0,
        text="Supply chain risk text.",
    ))
    row = db_conn.execute("SELECT text FROM prose_chunks").fetchone()
    assert row["text"] == "Supply chain risk text."
