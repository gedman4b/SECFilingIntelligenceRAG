"""
The Verifier is the trust boundary. Every check here corresponds to a specific failure mode from the previous system.
"""
import logging

from app.schemas import QueryPlan, Fact, Warning
from app.store.vector_store import ProsePassage
from app.instrumentation import log_latency
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

@log_latency(log)
def verify(
    question: str,
    plan: QueryPlan,
    facts: List[Fact],
    computed: Optional[float],
    prose: Optional[List[ProsePassage]] = None,
) -> Tuple[List[Warning], str]:
    warnings = []
    confidence = 'high'

    # Check 1: did the plan itself have ambiguity?
    if plan.ambiguity_flags:
        for flag in plan.ambiguity_flags:
            warnings.append(Warning(
                severity='warning',
                message=f'Question interpretation uncertain: {flag}',
            ))
        confidence = 'medium'

    # Check 2: was any requested fact missing?
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison'):
        if not facts:
            warnings.append(Warning(
                severity='error',
                message='No matching facts found in the corpus for this question.',
            ))
            log.info("question_type=%s -> confidence=insufficient_data (no facts)", plan.question_type)
            return warnings, 'insufficient_data'
 
    # Check 3: period alignment
    for period in plan.periods:
        matched = any(f.period.year == period.year
                     and f.period.quarter == period.quarter
                     for f in facts)
        if not matched:
            warnings.append(Warning(
                severity='error',
                message=f'Requested period Y{period.year} Q{period.quarter} not found.',
                field='period',
            ))
            confidence = 'insufficient_data'
 
    # Check 4: GAAP consistency
    if plan.gaap_preference == 'gaap':
        for f in facts:
            if not f.is_gaap:
                warnings.append(Warning(
                    severity='warning',
                    message=f'Retrieved value is non-GAAP but question implied GAAP.',
                    field='is_gaap',
                ))
                confidence = 'medium' if confidence == 'high' else confidence
 
    # Check 5: restated figures
    for f in facts:
        if f.is_restated:
            warnings.append(Warning(
                severity='info',
                message=f'Value at {f.period} is a restated figure.',
            ))
 
    # Check 6: metric canonicalization ambiguity
    for f in facts:
        if f.ambiguity_flags:
            warnings.append(Warning(
                severity='warning',
                message=f'Metric label matched with ambiguity: {", ".join(f.ambiguity_flags)}',
            ))
            confidence = 'medium' if confidence == 'high' else confidence
 
    # Check 7: sanity band on computed growth rates. Only applies to
    # growth_calc, where `computed` is a percentage. For comparison,
    # `compute()` returns a raw dollar delta (see numerical_reasoner.py),
    # and a multi-billion-dollar delta is normal, not a scale error; the
    # same numeric threshold does not mean the same thing for both.
    if plan.question_type == 'growth_calc' and computed is not None and abs(computed) > 5000:
        warnings.append(Warning(
            severity='warning',
            message=f'Computed growth rate of {computed:.1f}% is unusually large; ' +
                    'possible scale mismatch between periods.',
        ))
        confidence = 'low'

    # Check 8: narrative questions must retrieve at least one passage.
    # Without this, a failed prose search would silently reach the Answer
    # Composer with nothing to ground the narrative in.
    if plan.question_type == 'narrative':
        if not prose:
            warnings.append(Warning(
                severity='error',
                message='No relevant narrative passages found in the corpus for this question.',
            ))
            log.info("question_type=narrative -> confidence=insufficient_data (no passages)")
            return warnings, 'insufficient_data'

    # Check 9: the same metric+period sourced from more than one table.
    # SEC filings routinely restate primary-statement figures elsewhere
    # (MD&A commentary, footnotes, subsidiary/VIE disclosures that reuse a
    # generic label like "Total assets" for an unrelated, much smaller
    # scope), and the Fact Retriever's query has no preference logic on
    # its own -- it returns every matching row; is_preferred_source is
    # computed separately, from which table resolves the most distinct
    # canonical metrics (the true primary statement resolves many; a
    # footnote or MD&A table that incidentally repeats one figure resolves
    # only one or two).
    #
    # When every source agrees, that is extra corroborating evidence, not
    # a problem. When they disagree AND exactly one source is preferred,
    # trust it (agents/fact_retriever.resolve_preferred_facts() is what
    # every downstream stage actually computes/cites from) but disclose
    # the disagreement rather than hiding it. When they disagree with no
    # single preferred source (a tie, or no clear winner), there is no
    # principled way to pick one, and the system must fail closed rather
    # than silently citing whichever row SQLite happened to return first.
    grouped: dict = {}
    for f in facts:
        key = (f.metric_canonical_id, f.period.year, f.period.quarter)
        grouped.setdefault(key, []).append(f)
    for (metric_id, year, quarter), group in grouped.items():
        distinct_tables = {g.table_id for g in group}
        if len(distinct_tables) <= 1:
            continue
        distinct_values = {g.value for g in group}
        if len(distinct_values) > 1:
            preferred = [g for g in group if g.is_preferred_source]
            preferred_values = {g.value for g in preferred}
            if len(preferred_values) == 1:
                warnings.append(Warning(
                    severity='info',
                    message=(
                        f'{metric_id} Y{year}Q{quarter}: {len(distinct_tables) - len(preferred)} '
                        f'other source table(s) reported a different value (e.g. a subsidiary, '
                        f'VIE, or footnote disclosure reusing the same label); using the primary '
                        f'statement figure {next(iter(preferred_values)):,}.'
                    ),
                ))
            else:
                warnings.append(Warning(
                    severity='error',
                    message=(
                        f'{metric_id} Y{year}Q{quarter}: conflicting values across '
                        f'{len(distinct_tables)} source tables with no single preferred source: '
                        f'{sorted(distinct_values)}'
                    ),
                ))
                confidence = 'insufficient_data'
        else:
            warnings.append(Warning(
                severity='info',
                message=(
                    f'{metric_id} Y{year}Q{quarter}: corroborated by '
                    f'{len(distinct_tables)} source tables (values agree).'
                ),
            ))

    log.info(
        "question_type=%s, %d facts -> confidence=%s (%d warnings)",
        plan.question_type, len(facts), confidence, len(warnings),
    )
    return warnings, confidence
