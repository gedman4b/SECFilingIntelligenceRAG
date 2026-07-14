"""Tests for agents/fact_retriever.py. retrieve_facts() calls
store.db.get_conn() with no path argument, so these tests point it at a
temp file via the FACT_STORE_DB_PATH env var rather than the real store.

Deterministic except for the last-resort LLM-assisted synonym tier
(resolve_canonical_metric_via_llm, imported from ingest/canonicalizer.py):
any test that could reach it mocks it explicitly, per AGENTS.md ("No test
that requires network access to a live LLM provider runs in the default
suite")."""

from __future__ import annotations

import pytest

from app.agents import fact_retriever
from app.agents.fact_retriever import (
    resolve_preferred_facts, retrieve_facts, retrieve_ranking_facts, retrieve_ratio_facts,
)
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


def test_is_audited_derives_from_form_type(seeded_conn):
    """10-K figures are audited; 10-Q figures are not, per standard SEC
    practice. Deterministic from form_type, not an LLM judgment."""
    insert_filing(seeded_conn, FilingRecord(
        id="TSLA-10Q-2026-03-31", company_ticker="TSLA", form_type="10-Q",
        fiscal_year=2026, fiscal_quarter=1, period_end_date="2026-03-31",
        filing_url="file:///tsla-q.pdf", filing_path="/tsla-q.pdf",
    ))
    insert_fact(seeded_conn, _fact(quarter=None, value=94827.0))  # from the 10-K
    insert_fact(seeded_conn, _fact(
        row_id=2, quarter=1, year=2026, value=22387.0,
        filing_id="TSLA-10Q-2026-03-31",
        table_id="TSLA-10Q-2026-03-31::page5::table1",
    ))

    plan_10k = QueryPlan(question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
                          metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2025)])
    plan_10q = QueryPlan(question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
                          metric_canonical_id="METRIC_TOTAL_REVENUE", periods=[Period(year=2026, quarter=1)])

    facts_10k = retrieve_facts(plan_10k)
    facts_10q = retrieve_facts(plan_10q)
    assert facts_10k[0].is_audited is True
    assert facts_10q[0].is_audited is False


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


def test_retrieve_ratio_facts_gets_both_sides(seeded_conn):
    insert_fact(seeded_conn, _fact(
        row_id=1, metric_canonical_id="METRIC_GROSS_PROFIT",
        metric_raw_label="Gross profit", value=17094.0,
    ))
    insert_fact(seeded_conn, _fact(row_id=2, value=94827.0))  # METRIC_TOTAL_REVENUE
    plan = QueryPlan(
        question_type=QuestionType.MARGIN_CALC, company_ticker="TSLA",
        metric_canonical_id="METRIC_GROSS_PROFIT",
        ratio_denominator_canonical_id="METRIC_TOTAL_REVENUE",
        periods=[Period(year=2025)],
    )
    numerator, denominator = retrieve_ratio_facts(plan)
    assert len(numerator) == 1 and numerator[0].value == 17094.0
    assert len(denominator) == 1 and denominator[0].value == 94827.0


def test_retrieve_facts_falls_back_to_raw_label_when_unresolved(seeded_conn):
    """A balance-sheet line with no canonical registry entry -- e.g.
    Tesla's 'Digital assets' -- resolve_canonical_metric() returns None
    for it, but the fact was still extracted and stored
    (metric_canonical_id=UNRESOLVED, per ingest/canonicalizer.py). A
    question naming that exact phrase must still find it."""
    insert_fact(seeded_conn, _fact(
        metric_canonical_id="UNRESOLVED",
        metric_raw_label="Digital assets",
        value=1234.0,
        ambiguity_flags=["unrecognized_label: 'Digital assets' did not match the canonical metric registry"],
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="digital assets",
        periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert facts[0].value == 1234.0
    assert facts[0].ambiguity_flags  # carries the unresolved flag through


def test_retrieve_facts_raw_label_fallback_is_exact_match_not_fuzzy(seeded_conn, monkeypatch):
    """A near-miss phrase must not match at the raw-label tier -- no
    fuzzy or embedding matching on the numeric path, per the write-up's
    ban on embeddings for numeric queries. The LLM-assisted tier is
    mocked to also decline here (a plausible, honest outcome for this
    phrase), so this test is hermetic and doesn't depend on what a real
    model call would return."""
    monkeypatch.setattr(fact_retriever, "resolve_canonical_metric_via_llm", lambda phrase: None)
    insert_fact(seeded_conn, _fact(
        metric_canonical_id="UNRESOLVED",
        metric_raw_label="Digital assets",
        value=1234.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="digital asset holdings",  # not an exact match
        periods=[Period(year=2025)],
    )
    assert retrieve_facts(plan) == []


def test_retrieve_facts_falls_back_to_llm_synonym_when_raw_label_also_misses(seeded_conn, monkeypatch):
    """The real class of bug this tier exists for: a colloquial phrase
    like "the bottom line" resolves to nothing at tier 1 (not a curated
    synonym) and nothing at tier 2 (no filing literally uses that raw
    label), so tier 3 is the only way this question can be answered at
    all. Mocked here to isolate fact_retriever's own wiring from the
    model's actual judgment (that's ingest/canonicalizer.py's test
    surface instead)."""
    monkeypatch.setattr(fact_retriever, "resolve_canonical_metric_via_llm", lambda phrase: "METRIC_NET_INCOME")
    insert_fact(seeded_conn, _fact(
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income", value=3855.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="the bottom line",
        periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert facts[0].value == 3855.0


def test_llm_synonym_match_is_flagged_ambiguous_not_silently_trusted(seeded_conn, monkeypatch):
    """A match found via the LLM-assisted tier must never look as
    trustworthy as an exact registry match -- the Verifier's Check 6
    downgrades confidence based on exactly this flag."""
    monkeypatch.setattr(fact_retriever, "resolve_canonical_metric_via_llm", lambda phrase: "METRIC_NET_INCOME")
    insert_fact(seeded_conn, _fact(
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income", value=3855.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="the bottom line",
        periods=[Period(year=2025)],
    )
    facts = retrieve_facts(plan)
    assert len(facts) == 1
    assert any("llm_synonym_match" in flag for flag in facts[0].ambiguity_flags)


def test_llm_synonym_tier_never_called_when_exact_match_already_succeeded(seeded_conn, monkeypatch):
    """Cost and latency guard as much as a correctness one: the LLM tier
    must only run as a last resort, never when a cheaper deterministic
    tier already resolved the metric."""
    was_called = []
    monkeypatch.setattr(
        fact_retriever, "resolve_canonical_metric_via_llm",
        lambda phrase: was_called.append(phrase) or "METRIC_NET_INCOME",
    )
    insert_fact(seeded_conn, _fact(
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income", value=3855.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="net income",  # resolves at tier 1
        periods=[Period(year=2025)],
    )
    retrieve_facts(plan)
    assert was_called == []


def test_retrieve_ratio_facts_uses_llm_synonym_tier_for_either_side(seeded_conn, monkeypatch):
    """margin_calc's numerator and denominator are each resolved through
    the same three-tier cascade independently."""
    def fake_llm(phrase):
        return {"the bottom line": "METRIC_NET_INCOME", "top line": "METRIC_TOTAL_REVENUE"}.get(phrase)
    monkeypatch.setattr(fact_retriever, "resolve_canonical_metric_via_llm", fake_llm)
    insert_fact(seeded_conn, _fact(
        row_id=1, metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income", value=3855.0,
    ))
    insert_fact(seeded_conn, _fact(row_id=2, value=94827.0))  # METRIC_TOTAL_REVENUE
    plan = QueryPlan(
        question_type=QuestionType.MARGIN_CALC, company_ticker="TSLA",
        metric_natural_language="the bottom line",
        ratio_denominator_natural_language="top line",
        periods=[Period(year=2025)],
    )
    numerator, denominator = retrieve_ratio_facts(plan)
    assert len(numerator) == 1 and numerator[0].value == 3855.0
    assert len(denominator) == 1 and denominator[0].value == 94827.0


def test_retrieve_facts_does_not_fall_back_when_canonical_id_resolves_but_has_no_data(seeded_conn):
    """A canonical metric that resolved correctly but simply has no facts
    for the requested period is a real 'no data' case, not a raw-label
    mismatch -- must stay empty, not silently retry against unrelated
    stored labels."""
    plan = QueryPlan(
        question_type=QuestionType.NUMERIC_LOOKUP, company_ticker="TSLA",
        metric_natural_language="total revenue",  # resolves to METRIC_TOTAL_REVENUE
        periods=[Period(year=2099)],  # no data ingested for this year
    )
    assert retrieve_facts(plan) == []


def test_retrieve_ranking_facts_only_includes_metrics_with_both_periods(seeded_conn):
    insert_filing(seeded_conn, FilingRecord(
        id="TSLA-10K-2024-12-31", company_ticker="TSLA", form_type="10-K",
        fiscal_year=2024, period_end_date="2024-12-31",
        filing_url="file:///tsla-2024.pdf", filing_path="/tsla-2024.pdf",
    ))
    # METRIC_TOTAL_REVENUE has both 2024 and 2025 -> eligible
    insert_fact(seeded_conn, _fact(row_id=1, year=2024, value=97690.0, filing_id="TSLA-10K-2024-12-31",
                                    table_id="TSLA-10K-2024-12-31::page61::table1"))
    insert_fact(seeded_conn, _fact(row_id=2, year=2025, value=94827.0))
    # METRIC_SGA only has 2025 -> excluded
    insert_fact(seeded_conn, _fact(
        row_id=3, year=2025, metric_canonical_id="METRIC_SGA",
        metric_raw_label="Selling, general and administrative", value=5834.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.RANKING, company_ticker="TSLA",
        periods=[Period(year=2024), Period(year=2025)],
    )
    ranking_facts = retrieve_ranking_facts(plan)
    assert "METRIC_TOTAL_REVENUE" in ranking_facts
    assert "METRIC_SGA" not in ranking_facts
    assert len(ranking_facts["METRIC_TOTAL_REVENUE"]) == 2


def test_retrieve_ranking_facts_scope_expense_restricts_candidates(seeded_conn):
    insert_filing(seeded_conn, FilingRecord(
        id="TSLA-10K-2024-12-31", company_ticker="TSLA", form_type="10-K",
        fiscal_year=2024, period_end_date="2024-12-31",
        filing_url="file:///tsla-2024.pdf", filing_path="/tsla-2024.pdf",
    ))
    insert_fact(seeded_conn, _fact(row_id=1, year=2024, value=97690.0, filing_id="TSLA-10K-2024-12-31",
                                    table_id="TSLA-10K-2024-12-31::page61::table1"))
    insert_fact(seeded_conn, _fact(row_id=2, year=2025, value=94827.0))
    plan = QueryPlan(
        question_type=QuestionType.RANKING, company_ticker="TSLA", ranking_scope="expense",
        periods=[Period(year=2024), Period(year=2025)],
    )
    ranking_facts = retrieve_ranking_facts(plan)
    assert "METRIC_TOTAL_REVENUE" not in ranking_facts  # not an expense metric, scoped out


def test_retrieve_ranking_facts_requires_exactly_two_periods(seeded_conn):
    plan = QueryPlan(question_type=QuestionType.RANKING, company_ticker="TSLA", periods=[Period(year=2025)])
    assert retrieve_ranking_facts(plan) == {}


def test_retrieve_ratio_facts_empty_when_denominator_unresolved(seeded_conn):
    insert_fact(seeded_conn, _fact(
        row_id=1, metric_canonical_id="METRIC_GROSS_PROFIT",
        metric_raw_label="Gross profit", value=17094.0,
    ))
    plan = QueryPlan(
        question_type=QuestionType.MARGIN_CALC, company_ticker="TSLA",
        metric_canonical_id="METRIC_GROSS_PROFIT",
        # no ratio_denominator_* set at all
        periods=[Period(year=2025)],
    )
    numerator, denominator = retrieve_ratio_facts(plan)
    assert len(numerator) == 1
    assert denominator == []
