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
    defaults = dict(question_type=QuestionType.NUMERIC_LOOKUP, periods=[Period(year=2025)])
    defaults.update(overrides)
    return QueryPlan(**defaults)


# Check 1: plan-level ambiguity
def test_plan_ambiguity_downgrades_to_medium():
    warnings, confidence = verify("q", _plan(ambiguity_flags=["unclear period"]), [_fact()], None)
    assert confidence == "medium"
    assert any("interpretation uncertain" in w.message for w in warnings)


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
