# Deterministic. No LLM. This is the stage that prevents hallucination by construction.
from app.schemas import QueryPlan, Fact, Period
from app.store.db import get_conn
from app.ingest.canonicalizer import resolve_canonical_metric
from typing import List

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
        rows = conn.execute('''
            SELECT f.value, f.units, f.year, f.quarter, f.is_ttm,
                   f.metric_canonical_id, f.metric_raw_label,
                   f.is_gaap, f.is_restated,
                   f.filing_id, fi.filing_url, f.page_number,
                   f.table_id, f.row_id, f.ambiguity_flags
            FROM facts f JOIN filings fi ON fi.id = f.filing_id
            WHERE f.company_ticker = ?
              AND f.metric_canonical_id = ?
              AND f.year = ?
              AND (f.quarter IS ? OR f.quarter = ?)
              AND f.is_gaap = ?
        ''', (
            plan.company_ticker, metric_id,
            period.year, period.quarter, period.quarter,
            plan.gaap_preference != 'non_gaap',
        )).fetchall()
        for r in rows:
            results.append(Fact(
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
            ))
    return results
