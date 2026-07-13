"""Tests for agents/fact_retriever.py. Deterministic, no LLM: retrieve_facts()
calls store.db.get_conn() with no path argument, so these tests point it at
a temp file via the FACT_STORE_DB_PATH env var rather than the real store."""

from __future__ import annotations

import pytest

from app.agents.fact_retriever import resolve_preferred_facts, retrieve_facts
from app.schemas import Period, QueryPlan, QuestionType
from app.store.db import FactRecord, FilingRecord, get_conn, insert_fact, insert_filing


@pytest.fixture
def seeded_conn(tmp_path, monkeypatch):
    db_path = str(tmp_path / "fact_store.db")
    monkeypatch.setenv("FACT_STORE_DB_PATH", db_path)
    conn = get_conn(db_path)
    insert_filing(conn, FilingRecord(
        id="TSLA-10K-2025-12-31", company_ticker="TSLA", form_type="10-K",
        fiscal_year=2025, period_end_date="2025-12-31",
        filing_url="file:///tsla.pdf", filing_path="/tsla.pdf",
    ))
    yield conn
    conn.close()


def _fact(**overrides):
    defaults = dict(
        company_ticker="TSLA", metric_canonical_id="METRIC_TOTAL_REVENUE",
        metric_raw_label="Total revenues", value=94827.0, units="USD_millions",
        year=2025, filing_id="TSLA-10K-2025-12-31", page_number=61,
        table_id="TSLA-10K-2025-12-31::page61::table1", row_id=1,
    )
    defaults.update(overrides)
    return FactRecord(**defaults)


def test_retrieve_facts_basic_lookup(seeded_conn):
    insert_fact(seeded_conn, _fact())
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert facts[0].value == 94827.0
    assert facts[0].filing_url == "file:///tsla.pdf"


def test_retrieve_facts_returns_empty_when_no_metric_given():
    plan = QueryPlan(question_type=QuestionType.NUMERIC_LOOKUP, periods=[Period(year=2025)])
    assert retrieve_facts(plan) == []


def test_retrieve_facts_excludes_ytd_facts(seeded_conn):
    """Nothing in QueryPlan can request a YTD figure yet; it must never be
    silently substituted for a full-year or discrete-quarter request."""
    insert_fact(seeded_conn, _fact(
        row_id=2, quarter=2, is_ytd=True, value=219659.0,
        table_id="TSLA-10K-2025-12-31::page4::table1",
    ))
    insert_fact(seeded_conn, _fact(
        row_id=3, quarter=2, is_ytd=False, value=95359.0,
        table_id="TSLA-10K-2025-12-31::page4::table1",
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025, quarter=2)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert facts[0].value == 95359.0


def test_retrieve_facts_full_year_query_excludes_quarter_facts(seeded_conn):
    insert_fact(seeded_conn, _fact(quarter=None, value=94827.0))
    insert_fact(seeded_conn, _fact(row_id=2, quarter=1, value=22387.0))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert facts[0].value == 94827.0


def test_richer_table_is_marked_preferred_source(seeded_conn):
    """A table resolving more distinct canonical metrics is the true
    primary statement; a table resolving only one metric (e.g. MD&A
    restating a single figure) is not."""
    insert_fact(seeded_conn, _fact(
        row_id=1, table_id="TSLA-10K-2025-12-31::page61::table1", value=94827.0,
    ))
    insert_fact(seeded_conn, _fact(
        row_id=2, table_id="TSLA-10K-2025-12-31::page61::table1",
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income", value=3855.0,
    ))
    insert_fact(seeded_conn, _fact(
        row_id=3, table_id="TSLA-10K-2025-12-31::page45::table1", value=94827.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 2
    preferred = [f for f in facts if f.is_preferred_source]
    assert len(preferred) == 1
    assert preferred[0].page_number == 61


def test_resolve_preferred_facts_collapses_agreeing_duplicates(seeded_conn):
    insert_fact(seeded_conn, _fact(row_id=1, table_id="...page45::table1", value=94827.0))
    insert_fact(seeded_conn, _fact(row_id=2, table_id="...page61::table1", value=94827.0))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    resolved = resolve_preferred_facts(facts)
    assert len(resolved) == 1
    assert resolved[0].value == 94827.0


def test_resolve_preferred_facts_keeps_unresolved_ties(seeded_conn):
    """Two equally-rich tables with different values: no principled winner,
    both must survive resolution so the Verifier can fail closed on them."""
    insert_fact(seeded_conn, _fact(row_id=1, table_id="...page10::table1", value=100.0))
    insert_fact(seeded_conn, _fact(row_id=2, table_id="...page20::table1", value=200.0))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    resolved = resolve_preferred_facts(facts)
    assert len(resolved) == 2
