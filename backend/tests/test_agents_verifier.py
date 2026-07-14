"""Tests for agents/verifier.py, the trust boundary. Deterministic, no LLM.
Every check corresponds to a specific failure mode named in the write-up."""

from __future__ import annotations

from app.agents.verifier import verify
from app.schemas import Fact, Period, QueryPlan, QuestionType
from app.store.vector_store import ProsePassage


def _fact(**overrides) -> Fact:
    defaults = dict(
        value=100.0, units="USD_millions", period=Period(year=2025),
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income",
        is_gaap=True, filing_id="X", filing_url="u", page_number=1,
        table_id="X::p1::t1", row_id=1,
    )
    defaults.update(overrides)
    return Fact(**defaults)


def _plan(**overrides) -> QueryPlan:
    # company_ticker defaults to a real value so tests targeting OTHER
    # checks aren't incidentally short-circuited by Check 1b (missing
    # company); Check 1b has its own dedicated tests below that
    # explicitly omit it.
    defaults = dict(
        question_type=QuestionType.NUMERIC_LOOKUP, periods=[Period(year=2025)],
        company_ticker="TSLA",
    )
    defaults.update(overrides)
    return QueryPlan(**defaults)


# Check 1: plan-level ambiguity
def test_plan_ambiguity_downgrades_to_medium():
    warnings, confidence = verify("q", _plan(ambiguity_flags=["unclear period"]), [_fact()], None)
    assert confidence == "medium"
    assert any("interpretation uncertain" in w.message for w in warnings)


# Check 1b: a missing company_ticker fails closed deterministically,
# regardless of whether the Planner's own ambiguity_flags happened to
# mention it (LLM judgment, not guaranteed -- confirmed via live testing
# that the identical unambiguous no-company question was flagged on one
# run and not on another).
def test_missing_company_ticker_is_insufficient_data_for_numeric_question():
    plan = _plan(company_ticker=None)
    warnings, confidence = verify("q", plan, [], None)
    assert confidence == "insufficient_data"
    assert any("no company specified" in w.message.lower() for w in warnings)


def test_missing_company_ticker_is_insufficient_data_for_ranking():
    plan = _plan(question_type=QuestionType.RANKING, company_ticker=None)
    warnings, confidence = verify("q", plan, [], None)
    assert confidence == "insufficient_data"
    assert any("no company specified" in w.message.lower() for w in warnings)


def test_missing_company_ticker_does_not_block_narrative():
    """prose_retriever.py's vector search works fine with
    company_ticker=None -- it searches across every company -- so
    narrative must not be short-circuited by Check 1b the way the
    SQL-scoped question types are."""
    from app.store.vector_store import ProsePassage
    plan = _plan(question_type=QuestionType.NARRATIVE, company_ticker=None, periods=[])
    passages = [ProsePassage(
        chunk_id="c1", filing_id="X", section_type="risk_factors",
        page_start=1, page_end=1, text="risk text", distance=0.1,
    )]
    warnings, confidence = verify("q", plan, [], None, prose=passages)
    assert confidence != "insufficient_data"
    assert not any("no company specified" in w.message.lower() for w in warnings)


# Check 2: no facts for a numeric question fails closed
def test_no_facts_for_numeric_question_is_insufficient_data():
    warnings, confidence = verify("q", _plan(), [], None)
    assert confidence == "insufficient_data"
    assert any(w.severity == "error" for w in warnings)


def test_no_facts_is_fine_for_narrative_question():
    plan = _plan(question_type=QuestionType.NARRATIVE, periods=[])
    passages = [ProsePassage(
        chunk_id="c1", filing_id="X", section_type="risk_factors",
        page_start=1, page_end=1, text="risk text", distance=0.1,
    )]
    warnings, confidence = verify("q", plan, [], None, prose=passages)
    assert confidence == "high"


def test_narrative_with_a_resolved_period_is_not_blocked_by_check_3():
    """Regression guard for a real bug found via live testing: narrative
    questions never populate `facts` (they ground in `prose` instead),
    but the Planner routinely resolves relative-time phrases like "the
    previous quarter" into a concrete Period (planner.py rule 10). Check
    3 used to run unconditionally, so an always-empty `facts` list could
    never satisfy it -- "What did management cite as risks from the
    previous quarter for Tesla?" failed with "Requested period Y2026 Q1
    not found" despite good passages having been retrieved. Check 3 must
    be scoped to the question types that actually populate `facts`."""
    plan = _plan(question_type=QuestionType.NARRATIVE, periods=[Period(year=2026, quarter=1)])
    passages = [ProsePassage(
        chunk_id="c1", filing_id="X", section_type="risk_factors",
        page_start=1, page_end=1, text="risk text", distance=0.1,
    )]
    warnings, confidence = verify("q", plan, [], None, prose=passages)
    assert confidence == "high"
    assert not any("not found" in w.message for w in warnings)


# Check 3: period alignment
def test_missing_requested_period_is_insufficient_data():
    plan = _plan(periods=[Period(year=2025), Period(year=2024)])
    warnings, confidence = verify("q", plan, [_fact(period=Period(year=2025))], None)
    assert confidence == "insufficient_data"
    assert any("2024" in w.message for w in warnings)


# Check 4: GAAP consistency
def test_non_gaap_fact_downgrades_when_gaap_requested():
    plan = _plan(gaap_preference="gaap")
    warnings, confidence = verify("q", plan, [_fact(is_gaap=False)], None)
    assert confidence == "medium"
    assert any("non-GAAP" in w.message for w in warnings)


# Check 5: restated figures are informational only
def test_restated_fact_is_info_only_no_confidence_change():
    warnings, confidence = verify("q", _plan(), [_fact(is_restated=True)], None)
    assert confidence == "high"
    assert any(w.severity == "info" and "restated" in w.message for w in warnings)


# Check 6: canonicalization ambiguity on the fact itself
def test_fact_ambiguity_flags_downgrade_to_medium():
    warnings, confidence = verify("q", _plan(), [_fact(ambiguity_flags=["unrecognized_label: x"])], None)
    assert confidence == "medium"


def test_fact_ambiguity_flags_deduplicated_across_corroborating_sources():
    """Several source tables sharing the same unresolved raw label (a
    real case: Tesla's 'Automotive sales' reused across 4+ tables) must
    not each produce an identical warning."""
    facts = [
        _fact(table_id="X::p1::t1", ambiguity_flags=["unrecognized_label: 'Automotive sales' did not match the canonical metric registry"]),
        _fact(table_id="X::p2::t1", ambiguity_flags=["unrecognized_label: 'Automotive sales' did not match the canonical metric registry"]),
    ]
    warnings, confidence = verify("q", _plan(), facts, None)
    assert confidence == "medium"
    ambiguity_warnings = [w for w in warnings if "Metric label matched with ambiguity" in w.message]
    assert len(ambiguity_warnings) == 1


# Check 7: sanity band, growth_calc only
def test_extreme_growth_rate_downgrades_to_low():
    plan = _plan(question_type=QuestionType.GROWTH_CALC, periods=[Period(year=2024), Period(year=2025)])
    facts = [_fact(period=Period(year=2024)), _fact(period=Period(year=2025))]
    warnings, confidence = verify("q", plan, facts, 6000.0)
    assert confidence == "low"


def test_extreme_comparison_delta_does_not_trigger_sanity_band():
    """A multi-billion-dollar delta is normal for comparison (a raw dollar
    amount), not a percentage -- the same numeric threshold must not apply."""
    plan = _plan(question_type=QuestionType.COMPARISON, periods=[Period(year=2024), Period(year=2025)])
    facts = [_fact(period=Period(year=2024)), _fact(period=Period(year=2025))]
    warnings, confidence = verify("q", plan, facts, 6000.0)
    assert confidence == "high"


# Check 8: narrative must retrieve at least one passage
def test_narrative_with_no_passages_is_insufficient_data():
    plan = _plan(question_type=QuestionType.NARRATIVE, periods=[])
    warnings, confidence = verify("q", plan, [], None, prose=[])
    assert confidence == "insufficient_data"


# Check 9: multi-source agreement vs conflict
def test_agreeing_multi_source_facts_stay_high_confidence():
    facts = [
        _fact(table_id="X::p1::t1", value=100.0),
        _fact(table_id="X::p2::t1", value=100.0),
    ]
    warnings, confidence = verify("q", _plan(), facts, None)
    assert confidence == "high"
    assert any("corroborated" in w.message for w in warnings)


def test_conflicting_facts_with_clear_preferred_source_stay_high_confidence():
    facts = [
        _fact(table_id="X::p1::t1", value=137806.0, is_preferred_source=True),
        _fact(table_id="X::p2::t1", value=3134.0, is_preferred_source=False),
    ]
    warnings, confidence = verify("q", _plan(), facts, None)
    assert confidence == "high"
    assert any(w.severity == "info" and "primary statement figure" in w.message for w in warnings)


def test_conflicting_facts_with_no_clear_winner_fail_closed():
    facts = [
        _fact(table_id="X::p1::t1", value=100.0, is_preferred_source=True),
        _fact(table_id="X::p2::t1", value=200.0, is_preferred_source=True),
    ]
    warnings, confidence = verify("q", _plan(), facts, None)
    assert confidence == "insufficient_data"
    assert any(w.severity == "error" and "no single preferred source" in w.message for w in warnings)


# Check 10: unaudited figures (named explicitly in the assignment brief's
# ambiguity checklist)
def test_unaudited_fact_is_disclosed_as_info_not_downgraded():
    warnings, confidence = verify("q", _plan(), [_fact(is_audited=False)], None)
    assert confidence == "high"
    assert any(w.severity == "info" and "unaudited" in w.message for w in warnings)


def test_audited_fact_produces_no_unaudited_warning():
    warnings, confidence = verify("q", _plan(), [_fact(is_audited=True)], None)
    assert not any("unaudited" in w.message for w in warnings)


# Check 2b: margin_calc needs computed to be non-None, since Check 3's
# generic "some fact matches this period" logic can't tell whether both
# the numerator and denominator metric resolved.
def test_margin_calc_with_computed_value_is_not_insufficient():
    plan = _plan(question_type=QuestionType.MARGIN_CALC)
    warnings, confidence = verify("q", plan, [_fact(), _fact(row_id=2)], 18.03)
    assert confidence != "insufficient_data"


def test_margin_calc_without_computed_value_is_insufficient_even_with_facts():
    """Facts exist (e.g. only the numerator resolved), but computed is
    None because the denominator never resolved -- must still fail
    closed, not silently answer with only half the ratio."""
    plan = _plan(question_type=QuestionType.MARGIN_CALC)
    warnings, confidence = verify("q", plan, [_fact()], None)
    assert confidence == "insufficient_data"
    assert any("numerator and denominator" in w.message for w in warnings)


# Check 2c: ranking needs computed (a rank actually produced), same
# reasoning as margin_calc's Check 2b.
def test_ranking_with_no_facts_is_insufficient_data():
    plan = _plan(question_type=QuestionType.RANKING)
    warnings, confidence = verify("q", plan, [], None)
    assert confidence == "insufficient_data"


def test_ranking_with_facts_but_no_computed_is_insufficient_data():
    """Facts exist (some metric matched some period), but computed is None
    because no metric had a usable value in BOTH periods -- must still
    fail closed."""
    plan = _plan(question_type=QuestionType.RANKING)
    warnings, confidence = verify("q", plan, [_fact()], None)
    assert confidence == "insufficient_data"
    assert any("Could not rank" in w.message for w in warnings)


def test_ranking_with_computed_value_is_not_insufficient():
    plan = _plan(question_type=QuestionType.RANKING)
    warnings, confidence = verify("q", plan, [_fact(), _fact(row_id=2)], 50.0)
    assert confidence != "insufficient_data"
