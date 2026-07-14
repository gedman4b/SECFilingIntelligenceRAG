# Deterministic query and retrieval logic. The one exception, and the
# only LLM call anywhere in this module, is the last-resort synonym
# fallback in _resolve_metric_facts() -- see resolve_canonical_metric_via_llm()'s
# docstring in ingest/canonicalizer.py for why that's a different, safer
# mechanism than embeddings rather than a violation of "numeric queries
# never use embeddings."
import logging

from app.schemas import QueryPlan, Fact, Period
from app.store.db import get_conn
from app.ingest.canonicalizer import (
    resolve_canonical_metric, resolve_canonical_metric_via_llm,
    all_metric_ids, expense_metric_ids, normalize_label,
)
from app.instrumentation import log_latency
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _retrieve_facts_for_metric(
    company_ticker: Optional[str],
    metric_id: str,
    periods: List[Period],
    gaap_preference: str,
) -> List[Fact]:
    """Core retrieval: every fact for one already-resolved metric across
    the given periods, with is_audited and is_preferred_source computed.

    Shared by retrieve_facts() (one metric, the common case) and
    retrieve_ratio_facts() (two metrics, for margin_calc), so a margin
    question gets exactly the same YTD-exclusion, richness-ranking, and
    provenance handling as every other numeric path -- not a second,
    drifting implementation of the same query.
    """
    conn = get_conn()
    results: List[Fact] = []
    for period in periods:
        # is_ytd=0 excludes year-to-date cumulative facts (e.g. a 10-Q's
        # "six months ended" column). Nothing in QueryPlan can request a
        # YTD figure yet, so a YTD fact would otherwise collide with either
        # a full-year request (quarter IS NULL) or a discrete-quarter
        # request under the same quarter number, silently substituting a
        # partial-period value for the one actually asked for.
        #
        # table_richness ranks each source table by how many distinct
        # canonical metrics it resolves. Empirically, the true primary
        # financial statement resolves many metrics from one table, while
        # MD&A commentary or a footnote that happens to restate a single
        # figure resolves only one or two -- table_classifier.py cannot
        # otherwise distinguish "the income statement" from "MD&A restating
        # the income statement", since both get classified income_stmt.
        rows = conn.execute('''
            WITH table_richness AS (
                SELECT table_id, COUNT(DISTINCT metric_canonical_id) AS n_metrics
                FROM facts
                WHERE metric_canonical_id != 'UNRESOLVED'
                GROUP BY table_id
            )
            SELECT f.value, f.units, f.year, f.quarter, f.is_ttm,
                   f.metric_canonical_id, f.metric_raw_label,
                   f.is_gaap, f.is_restated,
                   f.filing_id, fi.filing_url, f.page_number,
                   f.table_id, f.row_id, f.ambiguity_flags, fi.form_type,
                   COALESCE(tr.n_metrics, 0) AS table_richness
            FROM facts f
            JOIN filings fi ON fi.id = f.filing_id
            LEFT JOIN table_richness tr ON tr.table_id = f.table_id
            WHERE f.company_ticker = ?
              AND f.metric_canonical_id = ?
              AND f.year = ?
              AND (f.quarter IS ? OR f.quarter = ?)
              AND f.is_gaap = ?
              AND f.is_ytd = 0
        ''', (
            company_ticker, metric_id,
            period.year, period.quarter, period.quarter,
            gaap_preference != 'non_gaap',
        )).fetchall()

        period_facts = [
            Fact(
                value=r['value'],
                units=r['units'],
                period=Period(year=r['year'], quarter=r['quarter'], is_ttm=r['is_ttm']),
                metric_canonical_id=r['metric_canonical_id'],
                metric_raw_label=r['metric_raw_label'],
                is_gaap=bool(r['is_gaap']),
                is_restated=bool(r['is_restated']),
                # 10-K/10-K-A annual statements are audited; 10-Q/10-Q-A
                # quarterly statements are not, per standard SEC practice.
                # A pure form_type lookup, not an LLM judgment.
                is_audited=r['form_type'] in ('10-K', '10-K/A'),
                filing_id=r['filing_id'],
                filing_url=r['filing_url'],
                page_number=r['page_number'],
                table_id=r['table_id'],
                row_id=r['row_id'],
                ambiguity_flags=(r['ambiguity_flags'] or '').split(',') if r['ambiguity_flags'] else [],
            )
            for r in rows
        ]
        if period_facts:
            richest = max(r['table_richness'] for r in rows)
            for fact, row in zip(period_facts, rows):
                fact.is_preferred_source = row['table_richness'] == richest
        results.extend(period_facts)
    return results


def _retrieve_facts_by_raw_label(
    company_ticker: Optional[str],
    raw_label_query: str,
    periods: List[Period],
    gaap_preference: str,
) -> List[Fact]:
    """Deterministic fallback for a metric that has no canonical registry
    entry -- e.g. a business-segment breakdown like "AWS revenue" or
    Tesla's "Energy generation and storage segment revenue", which
    consolidated financial-statement metrics (METRIC_TOTAL_REVENUE etc.)
    never carry. The canonical registry only covers consolidated
    statement line items on purpose (ingest/canonicalizer.py); segment and
    other one-off disclosures are still extracted and stored (as
    metric_canonical_id=UNRESOLVED, with an ambiguity_flag), just never
    forced into a canonical id.

    Matches by exact-match-after-normalization of the QUESTION's own
    metric text against each fact's stored metric_raw_label, using the
    identical normalization rule the canonical registry itself uses
    (ingest.canonicalizer.normalize_label) -- never a fuzzy or embedding
    match, per the numeric path's no-embeddings rule. This is a real
    second matching direction, not a duplicate of canonicalization at
    ingest time: ingestion matched each fact's raw label against the
    registry's known_labels; this matches the user's question text
    directly against facts' raw labels already in the store.

    Args:
        company_ticker: e.g. "TSLA".
        raw_label_query: The unresolved metric_natural_language text from
            the plan, e.g. "energy generation and storage segment revenue".
        periods: Periods to search.
        gaap_preference: Same GAAP filter as the canonical path.

    Returns:
        Matching facts, same shape and provenance fields as
        _retrieve_facts_for_metric(). Facts found this way keep whatever
        ambiguity_flags they were stored with (typically
        "unrecognized_label: ..."), which the Verifier's Check 6 already
        turns into a medium-confidence warning -- an accurate signal, since
        no registry entry vouches for this match the way a canonical
        metric_id would.
    """
    conn = get_conn()
    target = normalize_label(raw_label_query)
    results: List[Fact] = []
    for period in periods:
        rows = conn.execute('''
            WITH table_richness AS (
                SELECT table_id, COUNT(DISTINCT metric_canonical_id) AS n_metrics
                FROM facts
                WHERE metric_canonical_id != 'UNRESOLVED'
                GROUP BY table_id
            )
            SELECT f.value, f.units, f.year, f.quarter, f.is_ttm,
                   f.metric_canonical_id, f.metric_raw_label,
                   f.is_gaap, f.is_restated,
                   f.filing_id, fi.filing_url, f.page_number,
                   f.table_id, f.row_id, f.ambiguity_flags, fi.form_type,
                   COALESCE(tr.n_metrics, 0) AS table_richness
            FROM facts f
            JOIN filings fi ON fi.id = f.filing_id
            LEFT JOIN table_richness tr ON tr.table_id = f.table_id
            WHERE f.company_ticker = ?
              AND f.year = ?
              AND (f.quarter IS ? OR f.quarter = ?)
              AND f.is_gaap = ?
              AND f.is_ytd = 0
        ''', (
            company_ticker, period.year, period.quarter, period.quarter,
            gaap_preference != 'non_gaap',
        )).fetchall()

        matching_rows = [r for r in rows if normalize_label(r['metric_raw_label']) == target]
        period_facts = [
            Fact(
                value=r['value'],
                units=r['units'],
                period=Period(year=r['year'], quarter=r['quarter'], is_ttm=r['is_ttm']),
                metric_canonical_id=r['metric_canonical_id'],
                metric_raw_label=r['metric_raw_label'],
                is_gaap=bool(r['is_gaap']),
                is_restated=bool(r['is_restated']),
                is_audited=r['form_type'] in ('10-K', '10-K/A'),
                filing_id=r['filing_id'],
                filing_url=r['filing_url'],
                page_number=r['page_number'],
                table_id=r['table_id'],
                row_id=r['row_id'],
                ambiguity_flags=(r['ambiguity_flags'] or '').split(',') if r['ambiguity_flags'] else [],
            )
            for r in matching_rows
        ]
        if period_facts:
            richest = max(r['table_richness'] for r in matching_rows)
            for fact, row in zip(period_facts, matching_rows):
                fact.is_preferred_source = row['table_richness'] == richest
        results.extend(period_facts)
    return results


def _resolve_metric_facts(
    canonical_id: Optional[str],
    natural_language: Optional[str],
    company_ticker: Optional[str],
    periods: List[Period],
    gaap_preference: str,
) -> List[Fact]:
    """Three-tier metric resolution shared by retrieve_facts() (single
    metric) and retrieve_ratio_facts() (numerator and denominator, each
    resolved independently through this same cascade):

    1. Exact canonical match (resolve_canonical_metric) -- deterministic,
       free, the common case.
    2. The raw-label fallback (_retrieve_facts_by_raw_label) -- still
       exact-match, just against facts' own stored raw labels instead of
       the curated registry, for metrics like a business-segment
       breakdown with no registry entry at all.
    3. LLM-assisted synonym resolution (resolve_canonical_metric_via_llm)
       -- the last resort, only reached when both deterministic tiers
       above found nothing. Closed-set classification against the same
       registry, not fuzzy/embedding matching; see that function's
       docstring. Any match found this way is flagged as ambiguous on
       every returned Fact, so the Verifier downgrades confidence exactly
       like it already does for the raw-label fallback -- this tier never
       gets to claim the same certainty as an exact match.

    A canonical id that resolved but simply has no facts for the
    requested period does NOT fall through to tiers 2/3: that is a real
    "no data" case the Verifier must fail closed on, not a label-matching
    problem.
    """
    metric_id = canonical_id
    if not metric_id and natural_language:
        metric_id = resolve_canonical_metric(natural_language)

    if metric_id:
        return _retrieve_facts_for_metric(company_ticker, metric_id, periods, gaap_preference)

    if not natural_language:
        return []

    results = _retrieve_facts_by_raw_label(company_ticker, natural_language, periods, gaap_preference)
    if results:
        return results

    llm_metric_id = resolve_canonical_metric_via_llm(natural_language)
    if not llm_metric_id:
        return []

    results = _retrieve_facts_for_metric(company_ticker, llm_metric_id, periods, gaap_preference)
    flag = (
        f"llm_synonym_match: {natural_language!r} matched to {llm_metric_id} via "
        f"LLM-assisted synonym resolution, not an exact registry match"
    )
    for f in results:
        f.ambiguity_flags = f.ambiguity_flags + [flag]
    return results


@log_latency(log)
def retrieve_facts(plan: QueryPlan) -> List[Fact]:
    if not plan.metric_canonical_id and not plan.metric_natural_language:
        return []

    results = _resolve_metric_facts(
        plan.metric_canonical_id, plan.metric_natural_language,
        plan.company_ticker, plan.periods, plan.gaap_preference,
    )
    log.info(
        "plan(company=%s, metric=%s or %r, periods=%d) -> %d facts",
        plan.company_ticker, plan.metric_canonical_id, plan.metric_natural_language,
        len(plan.periods), len(results),
    )
    return results


@log_latency(log)
def retrieve_ratio_facts(plan: QueryPlan) -> Tuple[List[Fact], List[Fact]]:
    """Retrieve both sides of a margin_calc plan: numerator facts (the
    metric in metric_natural_language/metric_canonical_id, e.g. "gross
    profit") and denominator facts (ratio_denominator_*, e.g. "revenue").

    Args:
        plan: A QueryPlan with question_type=margin_calc.

    Returns:
        (numerator_facts, denominator_facts). Either list is empty if its
        metric could not be resolved or had no matching facts; the caller
        (main.py) is responsible for treating that as insufficient_data,
        same as retrieve_facts() returning [].
    """
    numerator_facts = _resolve_metric_facts(
        plan.metric_canonical_id, plan.metric_natural_language,
        plan.company_ticker, plan.periods, plan.gaap_preference,
    )
    denominator_facts = _resolve_metric_facts(
        plan.ratio_denominator_canonical_id, plan.ratio_denominator_natural_language,
        plan.company_ticker, plan.periods, plan.gaap_preference,
    )
    log.info(
        "plan(company=%s, numerator=%r, denominator=%r, periods=%d) -> %d/%d facts",
        plan.company_ticker, plan.metric_natural_language, plan.ratio_denominator_natural_language,
        len(plan.periods), len(numerator_facts), len(denominator_facts),
    )
    return numerator_facts, denominator_facts


@log_latency(log)
def retrieve_ranking_facts(plan: QueryPlan) -> Dict[str, List[Fact]]:
    """For question_type=ranking: every canonical metric (optionally scoped
    to expense-line metrics via plan.ranking_scope) that has a resolved
    fact in BOTH of plan.periods for plan.company_ticker.

    Ranking has no single target metric_canonical_id the way every other
    question_type does -- it scans the whole registry (or the expense
    subset) instead of resolving one metric_natural_language, so it needs
    its own retrieval entry point rather than reusing retrieve_facts().

    Args:
        plan: A QueryPlan with question_type=ranking and exactly two
            periods (the before/after periods being compared).

    Returns:
        Mapping from metric_canonical_id to its resolved facts, restricted
        to metrics with a preferred-source fact in both periods. A metric
        present in only one period is omitted entirely rather than ranked
        on a partial comparison.
    """
    if len(plan.periods) != 2:
        return {}

    candidate_ids = expense_metric_ids() if plan.ranking_scope == 'expense' else all_metric_ids()

    result: Dict[str, List[Fact]] = {}
    for metric_id in candidate_ids:
        facts = _retrieve_facts_for_metric(
            plan.company_ticker, metric_id, plan.periods, plan.gaap_preference,
        )
        resolved = resolve_preferred_facts(facts)
        periods_present = {(f.period.year, f.period.quarter) for f in resolved}
        if len(periods_present) == 2:
            result[metric_id] = resolved
    log.info(
        "plan(company=%s, scope=%s) -> %d/%d candidate metrics have both periods",
        plan.company_ticker, plan.ranking_scope, len(result), len(candidate_ids),
    )
    return result


def resolve_preferred_facts(facts: List[Fact]) -> List[Fact]:
    """Collapse multiple facts for the same (metric, period) down to the
    preferred source(s), for arithmetic and citation purposes.

    Two distinct source tables can report different values under the same
    raw label for the same metric and period -- not just a near-miss like
    "Net income" vs "Net income attributable to common stockholders" (a
    different canonical metric entirely, unaffected by this), but a
    genuine same-metric collision, e.g. a VIE or subsidiary footnote that
    also happens to use the generic label "Total assets" for a much
    smaller reporting scope than the consolidated balance sheet. Passing
    every source straight to numerical_reasoner.compute() risks silently
    using whichever one SQL happened to return, correct or not.

    This function does NOT decide whether a resolved conflict is safe to
    answer from; agents/verifier.py Check 9 does that by inspecting the
    original, unfiltered fact list (still required by every caller of
    retrieve_facts() for its conflict-detection accounting) and gating
    confidence accordingly. This function only picks which value(s) would
    be used if the answer proceeds.

    Args:
        facts: The full, unfiltered result of retrieve_facts().

    Returns:
        One fact per (metric, period) when a single source is clearly
        preferred (highest table_richness) or all sources agree. When
        multiple facts tie for preferred and disagree in value, all of
        them are returned unresolved -- there is no principled winner, and
        the Verifier will correctly gate on that ambiguity downstream.
    """
    groups: dict = {}
    for f in facts:
        key = (f.metric_canonical_id, f.period.year, f.period.quarter)
        groups.setdefault(key, []).append(f)

    resolved: List[Fact] = []
    for group in groups.values():
        preferred = [f for f in group if f.is_preferred_source]
        seen_values = set()
        for f in preferred:
            if f.value not in seen_values:
                resolved.append(f)
                seen_values.add(f.value)
    return resolved
