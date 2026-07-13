"""
eval/runner.py

Evaluation harness: runs eval/benchmark.py's cases against the query
pipeline and reports pass/fail per case.

Design principle
-----------------
AGENTS.md is explicit: "No test that requires network access to a live
LLM provider runs in the default suite. Mock at the client boundary." The
default run here never calls the Query Planner or Answer Composer (both
are LLM stages) -- it uses each case's hand-verified, golden QueryPlan
directly and exercises only the deterministic stages: Fact Retriever,
Numerical Reasoner, and the Verifier. This also matches AGENTS.md's
description of what a benchmark run should be: "Every agent has unit
tests using golden inputs and expected outputs." Narrative cases exercise
the Prose Retriever, which calls a local embedding model (onnxruntime),
not a live LLM provider, so it stays in the default run.

Because this harness bypasses the Planner, it cannot catch a Planner
misinterpretation; that risk is exactly why the Planner's prompt (not
tested here) needs separate, explicit review when it changes, per
AGENTS.md's "Every prompt change runs against the golden set before
merge." What this harness verifies is everything downstream of a correct
plan: does retrieval find the right value at the right page, does
arithmetic compute correctly, does the Verifier gate correctly.

Narrative correctness is not computed here as a pass/fail. Part 7 of the
write-up is explicit that narrative correctness is human-rated relevance
on a 1-3 scale, which no automated check can substitute for. A narrative
case is marked requires_human_review=True and reports whether retrieval
found anything at all and whether any expected_passage_keywords showed up
in the retrieved text, as a coarse sanity signal only.

This harness expects a fact store and vector index already populated by
the offline ingestion pipeline (ingest/pdf_parser.py through
ingest/canonicalizer.py) at the paths store.db and store.vector_store
resolve by default, or at --db-path / --chroma-dir if given. It does not
run ingestion itself, both to keep the default run LLM-free and because
ingestion is a one-time-per-filing offline job, not part of the query
pipeline this regression suite targets.

Dependencies
------------
pydantic >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

from pydantic import BaseModel, Field

from app.agents.fact_retriever import resolve_preferred_facts, retrieve_facts
from app.agents.numerical_reasoner import compute
from app.agents.prose_retriever import retrieve_prose
from app.agents.verifier import verify
from app.eval.benchmark import BENCHMARK, BenchmarkCase
from app.schemas import QuestionType

log = logging.getLogger(__name__)


# =============================================================================
# Result schema
# =============================================================================

class CaseResult(BaseModel):
    """Outcome of running one benchmark case."""
    case_id: str
    question_type: QuestionType
    passed: bool
    requires_human_review: bool = False
    details: List[str] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    """Summary of a full benchmark run."""
    results: List[CaseResult]
    total: int
    passed: int
    failed: int
    human_review_needed: int


# =============================================================================
# Public entry point
# =============================================================================

def run_benchmark(
    cases: Optional[List[BenchmarkCase]] = None,
    db_path: Optional[str] = None,
    chroma_dir: Optional[str] = None,
) -> BenchmarkReport:
    """Run every benchmark case against the deterministic query pipeline.

    Args:
        cases: Cases to run. Defaults to eval.benchmark.BENCHMARK.
        db_path: Override the SQLite fact store path for this run (sets
            FACT_STORE_DB_PATH for the duration, then restores it).
        chroma_dir: Override the Chroma persistence directory for this run
            (sets CHROMA_PERSIST_DIR for the duration, then restores it).

    Returns:
        The full report: one CaseResult per case plus pass/fail totals.
    """
    cases = BENCHMARK if cases is None else cases

    with _temporary_env("FACT_STORE_DB_PATH", db_path), \
         _temporary_env("CHROMA_PERSIST_DIR", chroma_dir):
        results = [_run_case(case) for case in cases]

    passed = sum(1 for r in results if r.passed)
    human_review = sum(1 for r in results if r.requires_human_review)
    report = BenchmarkReport(
        results=results, total=len(results), passed=passed,
        failed=len(results) - passed, human_review_needed=human_review,
    )
    log.info(
        "Benchmark: %d/%d passed, %d require human relevance review",
        passed, report.total, human_review,
    )
    return report


# =============================================================================
# Per-case execution
# =============================================================================

def _run_case(case: BenchmarkCase) -> CaseResult:
    """Dispatch one case to its numeric or narrative runner.

    A case that raises is recorded as a failure with the exception message
    rather than aborting the whole run, so one bad case does not hide the
    results of the other nineteen.
    """
    try:
        if case.question_type == QuestionType.NARRATIVE:
            return _run_narrative_case(case)
        return _run_numeric_case(case)
    except Exception as exc:
        log.error("Case %s raised: %s", case.case_id, exc)
        return CaseResult(
            case_id=case.case_id, question_type=case.question_type,
            passed=False, details=[f"runner exception: {exc}"],
        )


def _run_numeric_case(case: BenchmarkCase) -> CaseResult:
    """Run a numeric_lookup, growth_calc, or comparison case.

    Exercises Fact Retriever, Numerical Reasoner (if applicable), and the
    Verifier directly against case.plan -- the Query Planner is never
    called. Checks value, units, page (provenance), computed result, and
    the Verifier's confidence output.
    """
    details: List[str] = []
    facts = retrieve_facts(case.plan)
    resolved_facts = resolve_preferred_facts(facts)

    computed: Optional[float] = None
    if case.question_type in (QuestionType.GROWTH_CALC, QuestionType.COMPARISON):
        computed, _expr = compute(case.plan, resolved_facts)

    # verify() receives the raw, unfiltered facts -- matching main.py,
    # since Check 9's conflict accounting needs every source, not just the
    # resolved winner.
    _warnings, confidence = verify(case.question, case.plan, facts, computed)

    value_ok = True
    provenance_ok = True
    for expected in case.expected_facts:
        # Multiple tables can agree on the same (metric, period); cite the
        # one fact_retriever.py flagged as the primary source rather than
        # whichever SQL happened to return first.
        candidates = [
            f for f in facts
            if f.metric_canonical_id == expected.metric_canonical_id
            and f.period.year == expected.year
            and f.period.quarter == expected.quarter
        ]
        match = next((f for f in candidates if f.is_preferred_source), None) or (
            candidates[0] if candidates else None
        )
        if match is None:
            value_ok = False
            provenance_ok = False
            details.append(
                f"missing expected fact: {expected.metric_canonical_id} "
                f"Y{expected.year}Q{expected.quarter}"
            )
            continue
        if not math.isclose(match.value, expected.expected_value, rel_tol=1e-9):
            value_ok = False
            details.append(
                f"value mismatch for {expected.metric_canonical_id} "
                f"Y{expected.year}: got {match.value}, expected {expected.expected_value}"
            )
        if match.units != expected.expected_units:
            value_ok = False
            details.append(
                f"units mismatch for {expected.metric_canonical_id} "
                f"Y{expected.year}: got {match.units!r}, expected {expected.expected_units!r}"
            )
        if match.page_number != expected.expected_page:
            provenance_ok = False
            details.append(
                f"page mismatch for {expected.metric_canonical_id} "
                f"Y{expected.year}: got page {match.page_number}, expected {expected.expected_page}"
            )

    computed_ok = True
    if case.expected_computed_value is not None:
        if computed is None or abs(computed - case.expected_computed_value) > case.expected_computed_tolerance:
            computed_ok = False
            details.append(f"computed value mismatch: got {computed}, expected {case.expected_computed_value}")

    confidence_ok = confidence == case.expected_confidence
    if not confidence_ok:
        details.append(f"confidence mismatch: got {confidence!r}, expected {case.expected_confidence!r}")

    passed = value_ok and provenance_ok and computed_ok and confidence_ok
    return CaseResult(
        case_id=case.case_id, question_type=case.question_type,
        passed=passed, details=details,
    )


def _run_narrative_case(case: BenchmarkCase) -> CaseResult:
    """Run a narrative case.

    Only checks that retrieval found something and, as a coarse sanity
    signal, whether expected_passage_keywords appear in the retrieved
    text. Real correctness for narrative questions is human-rated
    relevance (Part 7 of the write-up), which this function does not and
    cannot compute; every narrative case is flagged requires_human_review.
    """
    passages = retrieve_prose(case.plan, case.question)
    details = [f"retrieved {len(passages)} passages"]

    matched = [
        kw for kw in case.expected_passage_keywords
        if any(kw.lower() in p.text.lower() for p in passages)
    ]
    if case.expected_passage_keywords:
        details.append(
            f"keyword sanity check: matched {matched} of {case.expected_passage_keywords} "
            "(informational only, not a substitute for human relevance rating)"
        )

    return CaseResult(
        case_id=case.case_id, question_type=case.question_type,
        passed=len(passages) > 0,
        requires_human_review=True,
        details=details,
    )


# =============================================================================
# Helpers
# =============================================================================

class _temporary_env:
    """Context manager: set an env var for a block, then restore it.

    Used to point the already-built store.db / store.vector_store
    connection helpers (which read FACT_STORE_DB_PATH / CHROMA_PERSIST_DIR
    with no other injection point) at a specific benchmark fixture without
    modifying those modules.
    """

    def __init__(self, name: str, value: Optional[str]) -> None:
        self._name = name
        self._value = value
        self._prior: Optional[str] = None

    def __enter__(self) -> None:
        if self._value is not None:
            self._prior = os.environ.get(self._name)
            os.environ[self._name] = self._value

    def __exit__(self, *exc_info) -> None:
        if self._value is not None:
            if self._prior is None:
                os.environ.pop(self._name, None)
            else:
                os.environ[self._name] = self._prior


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Run the hand-verified benchmark against a populated fact store.",
    )
    ap.add_argument(
        "--db-path", default=None,
        help="SQLite fact store to test against (defaults to FACT_STORE_DB_PATH / store/fact_store.db)",
    )
    ap.add_argument(
        "--chroma-dir", default=None,
        help="Chroma persistence directory (defaults to CHROMA_PERSIST_DIR / store/chroma_data)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = run_benchmark(db_path=args.db_path, chroma_dir=args.chroma_dir)

    for result in report.results:
        if result.passed:
            status = "PASS"
        elif result.requires_human_review:
            status = "REVIEW"
        else:
            status = "FAIL"
        print(f"[{status}] {result.case_id} ({result.question_type.value})")
        for line in result.details:
            print(f"    {line}")

    print(
        f"\n{report.passed}/{report.total} passed, "
        f"{report.human_review_needed} require human relevance review"
    )
    if report.failed > 0:
        sys.exit(1)
