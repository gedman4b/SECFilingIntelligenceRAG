"""Structural tests for eval/benchmark.py itself. Does not require the
real persistent fact store -- eval/runner.py's full run against real
ingested data is a separate, manually-triggered regression gate, not part
of this unit suite (see eval/runner.py's own docstring)."""

from __future__ import annotations

from collections import Counter

from app.eval.benchmark import BENCHMARK
from app.schemas import QuestionType


def test_benchmark_composition_matches_writeup():
    counts = Counter(c.question_type for c in BENCHMARK)
    assert counts[QuestionType.NUMERIC_LOOKUP] == 10
    assert counts[QuestionType.GROWTH_CALC] == 6
    assert counts[QuestionType.COMPARISON] == 2
    assert counts[QuestionType.NARRATIVE] == 2


def test_benchmark_case_ids_are_unique():
    ids = [c.case_id for c in BENCHMARK]
    assert len(ids) == len(set(ids))


def test_every_numeric_case_has_a_golden_plan_with_periods():
    for case in BENCHMARK:
        if case.question_type != QuestionType.NARRATIVE:
            assert case.plan.periods, f"{case.case_id} has no periods in its golden plan"


def test_growth_and_comparison_cases_have_two_periods():
    for case in BENCHMARK:
        if case.question_type in (QuestionType.GROWTH_CALC, QuestionType.COMPARISON):
            assert len(case.plan.periods) == 2, case.case_id


def test_narrative_cases_have_no_expected_facts():
    for case in BENCHMARK:
        if case.question_type == QuestionType.NARRATIVE:
            assert case.expected_facts == []
