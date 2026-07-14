"""Tests for agents/numerical_reasoner.py. Deterministic Python arithmetic,
never delegated to an LLM."""

from __future__ import annotations

from app.agents.numerical_reasoner import compute
from app.schemas import Fact, Period, QueryPlan, QuestionType


def _fact(year: int, value: float, quarter=None) -> Fact:
    return Fact(
        value=value, units="USD_millions", period=Period(year=year, quarter=quarter),
        metric_canonical_id="METRIC_NET_INCOME", metric_raw_label="Net income",
        is_gaap=True, filing_id="X", filing_url="u", page_number=1,
        table_id="X::p1::t1", row_id=1,
    )


def _metric_fact(metric_id: str, label: str, year: int, value: float) -> Fact:
    return Fact(
        value=value, units="USD_millions", period=Period(year=year),
        metric_canonical_id=metric_id, metric_raw_label=label,
        is_gaap=True, filing_id="X", filing_url="u", page_number=1,
        table_id="X::p1::t1", row_id=1,
    )


def test_growth_calc_matches_independently_verified_value():
    """Tesla net income FY2024->FY2025, independently computed by hand:
    ((3855-7153)/|7153|)*100 = -46.106529%, rounds to -46.11."""
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, expr = compute(plan, [_fact(2024, 7153.0), _fact(2025, 3855.0)])
    assert computed == -46.11
    assert "7,153" in expr and "3,855" in expr


def test_growth_calc_positive_growth():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(plan, [_fact(2023, 96773.0), _fact(2024, 97690.0)])
    assert computed == 0.95


def test_growth_calc_undefined_when_base_is_zero():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, message = compute(plan, [_fact(2024, 0.0), _fact(2025, 100.0)])
    assert computed is None
    assert "undefined" in message


def test_growth_calc_needs_at_least_two_facts():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, expr = compute(plan, [_fact(2025, 100.0)])
    assert computed is None and expr is None


def test_growth_calc_sorts_by_period_regardless_of_input_order():
    """The newer period must be treated as the endpoint even if it comes
    first in the input list."""
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(plan, [_fact(2025, 3855.0), _fact(2024, 7153.0)])
    assert computed == -46.11


def test_comparison_returns_raw_delta_not_percentage():
    """Independently verified: 54941-48390 = 6551."""
    plan = QueryPlan(question_type=QuestionType.COMPARISON)
    computed, expr = compute(plan, [_fact(2024, 48390.0), _fact(2025, 54941.0)])
    assert computed == 6551.0
    assert "%" not in expr


def test_comparison_delta_can_be_negative():
    plan = QueryPlan(question_type=QuestionType.COMPARISON)
    computed, _expr = compute(plan, [_fact(2024, 364980.0), _fact(2025, 359241.0)])
    assert computed == -5739.0


def test_narrative_and_numeric_lookup_never_compute():
    for qtype in (QuestionType.NARRATIVE, QuestionType.NUMERIC_LOOKUP, QuestionType.UNKNOWN):
        plan = QueryPlan(question_type=qtype)
        computed, expr = compute(plan, [_fact(2024, 100.0), _fact(2025, 200.0)])
        assert computed is None and expr is None


# =============================================================================
# margin_calc: a ratio between two DIFFERENT metrics in the same period,
# not one metric across two periods.
# =============================================================================

def test_margin_calc_matches_independently_verified_value():
    """Tesla FY2025 gross margin, independently computed: 17094/94827*100
    = 18.026...%, rounds to 18.03."""
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2025, 17094.0)]
    denominator = [_fact(2025, 94827.0)]
    computed, expr = compute(plan, numerator, denominator)
    assert computed == 18.03
    assert "17,094" in expr and "94,827" in expr


def test_margin_calc_uses_most_recent_common_period():
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2023, 100.0), _fact(2025, 200.0)]
    denominator = [_fact(2023, 1000.0), _fact(2025, 1000.0)]
    computed, expr = compute(plan, numerator, denominator)
    assert computed == 20.0  # 2025's ratio, not 2023's (10.0)
    assert "Y2025" in expr


def test_margin_calc_requires_both_sides():
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    computed, expr = compute(plan, [_fact(2025, 100.0)], [])
    assert computed is None


def test_margin_calc_requires_a_common_period():
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2023, 100.0)]
    denominator = [_fact(2024, 1000.0)]
    computed, message = compute(plan, numerator, denominator)
    assert computed is None
    assert "no period has both" in message


def test_margin_calc_undefined_when_denominator_is_zero():
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    computed, message = compute(plan, [_fact(2025, 100.0)], [_fact(2025, 0.0)])
    assert computed is None
    assert "undefined" in message


def test_margin_calc_computes_every_requested_period_not_just_latest():
    """Regression guard for a real bug found in live testing: given only
    the latest period's ratio plus raw facts for an earlier period, the
    Composer "helpfully" computed the earlier ratio itself -- exactly the
    arithmetic it must never do. Every common period must be computed
    here so the Composer has nothing left to fill in.

    Tesla FY2023/FY2024 gross margin, independently computed:
    17660/96773*100 = 18.2497...% (18.25); 17450/97690*100 = 17.858...% (17.86)."""
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2023, 17660.0), _fact(2024, 17450.0)]
    denominator = [_fact(2023, 96773.0), _fact(2024, 97690.0)]
    computed, expr = compute(plan, numerator, denominator)
    assert computed == 17.86  # most recent period
    assert "18.25%" in expr  # but 2023's ratio is also fully computed
    assert "17.86%" in expr
    assert "Y2023" in expr and "Y2024" in expr


def test_margin_calc_includes_precomputed_delta_between_periods():
    """Regression guard for a real bug found in live testing: given two
    fully-computed per-period ratios but no pre-computed delta between
    them, the Composer "helpfully" subtracted them itself ("an increase
    of 0.88 percentage points") -- still forbidden arithmetic, even
    though both inputs were already given. The delta must be computed
    here so nothing is left to compute.

    Tesla FY2024/FY2025 SG&A-as-%-of-revenue, independently verified:
    5150/97690*100=5.27%, 5834/94827*100=6.15%, delta=+0.88pp."""
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2024, 5150.0), _fact(2025, 5834.0)]
    denominator = [_fact(2024, 97690.0), _fact(2025, 94827.0)]
    computed, expr = compute(plan, numerator, denominator)
    assert computed == 6.15
    assert "+0.88 percentage points" in expr
    assert "Change from Y2024QFY to Y2025QFY" in expr


def test_margin_calc_no_delta_line_for_a_single_period():
    plan = QueryPlan(question_type=QuestionType.MARGIN_CALC)
    numerator = [_fact(2025, 17094.0)]
    denominator = [_fact(2025, 94827.0)]
    _computed, expr = compute(plan, numerator, denominator)
    assert "Change from" not in expr


# =============================================================================
# ranking: multiple metrics ranked by magnitude of change between two
# periods, not one named metric across two periods.
# =============================================================================

def test_ranking_top_increase_sorts_by_raw_growth_descending():
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="top_increase")
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 100.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 150.0)],  # +50%
        "METRIC_TOTAL_REVENUE": [
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2024, 1000.0),
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2025, 1100.0),
        ],  # +10%
    }
    computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed == 50.0
    assert expr.startswith("1. SG&A")


def test_ranking_top_decrease_sorts_ascending():
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="top_decrease")
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 100.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 90.0)],  # -10%
        "METRIC_TOTAL_REVENUE": [
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2024, 1000.0),
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2025, 500.0),
        ],  # -50%
    }
    computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed == -50.0
    assert expr.startswith("1. Total revenues")


def test_ranking_most_deteriorated_treats_rising_expense_and_falling_revenue_as_bad():
    """A rising expense (badness=+pct) and a falling revenue/profit metric
    (badness=-pct) must both be scored as deterioration, not compared on
    raw growth % alone -- a plain top_increase/top_decrease sort would
    wrongly favor whichever had the larger magnitude regardless of
    direction-of-harm."""
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="most_deteriorated")
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 100.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 150.0)],  # +50% growth, expense -> badness +50
        "METRIC_TOTAL_REVENUE": [
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2024, 1000.0),
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2025, 800.0),
        ],  # -20% growth, revenue -> badness +20
    }
    computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed == 50.0  # SG&A's raw growth_pct, still the worse-scored metric
    assert expr.startswith("1. SG&A")


def test_ranking_most_improved_is_the_reverse_of_most_deteriorated():
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="most_improved")
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 100.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 80.0)],  # expense fell 20% -> improvement
        "METRIC_TOTAL_REVENUE": [
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2024, 1000.0),
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2025, 900.0),
        ],  # revenue fell 10% -> deterioration
    }
    computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed == -20.0  # SG&A's raw growth_pct: the most-improved metric
    assert expr.startswith("1. SG&A")


def test_ranking_skips_zero_base_metric_without_failing_whole_ranking():
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="top_increase")
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 0.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 100.0)],  # undefined growth
        "METRIC_TOTAL_REVENUE": [
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2024, 1000.0),
            _metric_fact("METRIC_TOTAL_REVENUE", "Total revenues", 2025, 1100.0),
        ],
    }
    computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed == 10.0
    assert "SG&A" not in expr


def test_ranking_empty_candidates_dict_is_none_none():
    """No metric even had both periods -- retrieve_ranking_facts() already
    returned {}, so there's nothing to distinguish from the generic
    "no data" case; the Verifier's Check 2 (empty facts) covers this."""
    plan = QueryPlan(question_type=QuestionType.RANKING)
    computed, message = compute(plan, [], metric_facts_by_id={})
    assert computed is None and message is None


def test_ranking_all_candidates_zero_base_is_none_with_a_reason():
    """Every candidate metric had both periods, but every one had a zero
    base value -- a different failure mode from the dict being empty."""
    plan = QueryPlan(question_type=QuestionType.RANKING)
    metric_facts_by_id = {
        "METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 0.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 100.0)],
    }
    computed, message = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert computed is None
    assert "no candidate metric" in message


def test_ranking_caps_at_top_n():
    plan = QueryPlan(question_type=QuestionType.RANKING, ranking_direction="top_increase")
    metric_facts_by_id = {
        f"METRIC_{i}": [_metric_fact(f"METRIC_{i}", f"Metric {i}", 2024, 100.0), _metric_fact(f"METRIC_{i}", f"Metric {i}", 2025, 100.0 + i)]
        for i in range(1, 8)  # 7 candidates, more than RANKING_TOP_N=5
    }
    _computed, expr = compute(plan, [], metric_facts_by_id=metric_facts_by_id)
    assert expr.count(";") == 4  # 5 lines joined by "; " -> 4 separators


def test_ranking_ignores_metric_facts_by_id_for_other_question_types():
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(
        plan, [_fact(2024, 7153.0), _fact(2025, 3855.0)],
        metric_facts_by_id={"METRIC_SGA": [_metric_fact("METRIC_SGA", "SG&A", 2024, 1.0), _metric_fact("METRIC_SGA", "SG&A", 2025, 999.0)]},
    )
    assert computed == -46.11


def test_margin_calc_ignores_denominator_facts_for_other_question_types():
    """denominator_facts must be inert for every other question_type, so
    passing it defensively from main.py can never change growth_calc or
    comparison behavior."""
    plan = QueryPlan(question_type=QuestionType.GROWTH_CALC)
    computed, _expr = compute(
        plan, [_fact(2024, 7153.0), _fact(2025, 3855.0)], denominator_facts=[_fact(2025, 999.0)],
    )
    assert computed == -46.11
