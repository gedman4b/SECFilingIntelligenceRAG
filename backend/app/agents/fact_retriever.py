# Deterministic. No LLM. This is the stage that prevents hallucination by construction.
import logging

from app.schemas import QueryPlan, Fact, Period
from app.store.db import get_conn
from app.ingest.canonicalizer import resolve_canonical_metric
from app.instrumentation import log_latency
from typing import List

log = logging.getLogger(__name__)

@log_latency(log)
def retrieve_facts(plan: QueryPlan) -> List[Fact]:
    if not plan.metric_canonical_id and not plan.metric_natural_language:
        return []
 
    # Resolve canonical ID if needed
    metric_id = plan.metric_canonical_id
    if not metric_id and plan.metric_natural_language:
        metric_id = resolve_canonical_metric(plan.metric_natural_language)
 
    if not metric_id:
        return []
 
    conn = get_conn()
    results = []
    for period in plan.periods:
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
                   f.table_id, f.row_id, f.ambiguity_flags,
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
            plan.company_ticker, metric_id,
            period.year, period.quarter, period.quarter,
            plan.gaap_preference != 'non_gaap',
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
    log.info(
        "plan(company=%s, metric=%s, periods=%d) -> %d facts",
        plan.company_ticker, metric_id, len(plan.periods), len(results),
    )
    return results


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
