"""
The Verifier is the trust boundary. Every check here corresponds to a specific failure mode from the previous system.
"""
import logging

from app.schemas import QueryPlan, Fact, Warning
from app.store.vector_store import ProsePassage
from app.instrumentation import log_latency
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)


def _format_period(period) -> str:
    """Render a Period as "Y2025" or "Y2025 Q1" instead of its Pydantic
    repr, for warning messages a user actually reads."""
    return f"Y{period.year}" + (f" Q{period.quarter}" if period.quarter else "")


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

    # Check 1b: numeric_lookup/growth_calc/comparison/margin_calc/ranking
    # are all scoped by company_ticker in the Fact Retriever's SQL -- with
    # no ticker, the query can only ever return zero rows, which Check 2
    # below would otherwise report as an opaque "no matching facts found"
    # with no hint that the real, fixable problem is "you didn't name a
    # company". This is a plain, deterministic field check, not a re-use
    # of the Planner's own ambiguity_flags: live testing showed the same
    # unambiguous no-company question sometimes got an ambiguity_flag from
    # the Planner and sometimes didn't (LLM judgment, not guaranteed), so
    # relying on it alone was not reliable enough for something this
    # fundamental. narrative is exempt: prose_retriever.py's vector search
    # works fine with company_ticker=None (searches across every company).
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison', 'margin_calc', 'ranking'):
        if not plan.company_ticker:
            warnings.append(Warning(
                severity='error',
                message='No company specified. Please name a company (e.g. "Tesla", "Apple") in your question.',
                field='company_ticker',
            ))
            log.info("question_type=%s -> confidence=insufficient_data (no company_ticker)", plan.question_type)
            return warnings, 'insufficient_data'

    # Check 2: was any requested fact missing? For margin_calc, `facts`
    # is the combined numerator+denominator list main.py builds before
    # calling verify(); either side being completely absent means this
    # check already covers it (an empty combined list can only happen if
    # BOTH sides were empty, but if only ONE side resolved, Check 2 would
    # miss it -- see the margin-specific check just below instead).
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison', 'margin_calc', 'ranking'):
        if not facts:
            warnings.append(Warning(
                severity='error',
                message='No matching facts found in the corpus for this question.',
            ))
            log.info("question_type=%s -> confidence=insufficient_data (no facts)", plan.question_type)
            return warnings, 'insufficient_data'

    # Check 2b: margin_calc needs BOTH the numerator and denominator metric
    # resolved for the same period, not just "some fact exists" -- Check 3
    # below only verifies each requested period matches ANY fact in the
    # combined numerator+denominator list, which would wrongly look
    # satisfied if e.g. revenue resolved but gross profit did not.
    # numerical_reasoner.compute() already returns computed=None for every
    # margin failure mode (missing side, no common period, zero
    # denominator), so it is the one signal that actually distinguishes
    # "ratio computable" from not.
    if plan.question_type == 'margin_calc' and computed is None:
        warnings.append(Warning(
            severity='error',
            message='Could not compute the requested ratio: the numerator and denominator metrics were not both available for the same period.',
        ))
        log.info("question_type=margin_calc -> confidence=insufficient_data (computed is None)")
        return warnings, 'insufficient_data'

    # Check 2c: ranking needs at least one candidate metric with a usable
    # value in both requested periods. `facts` being non-empty (Check 2)
    # only means SOME metric had a fact in SOME period; computed is None
    # is the actual signal from numerical_reasoner.compute() that no
    # metric had both periods with a non-zero base.
    if plan.question_type == 'ranking' and computed is None:
        warnings.append(Warning(
            severity='error',
            message='Could not rank any metric: none had a usable value in both requested periods.',
        ))
        log.info("question_type=ranking -> confidence=insufficient_data (computed is None)")
        return warnings, 'insufficient_data'


    # Check 3: period alignment. `facts` is only ever populated for the
    # numeric-family question types (Stage B/C in main.py) -- narrative
    # questions ground their answer in `prose` instead (Check 8 covers
    # that) and never populate `facts` at all. Without this guard, any
    # narrative question where the Planner resolved a relative-time
    # phrase into a concrete period (e.g. "the previous quarter", per
    # planner.py rule 10) would always fail here, since an always-empty
    # `facts` list can never contain a matching period -- confirmed via
    # live testing: "What did management cite as risks from the previous
    # quarter for Tesla?" failed with "Requested period Y2026 Q1 not
    # found" despite prose retrieval never having been given a chance to
    # run first.
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison', 'margin_calc', 'ranking'):
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
 
    # Check 5: restated figures. Deduplicated per (metric, period): several
    # corroborating source tables for the same fact would otherwise each
    # produce an identical warning.
    seen_restated = set()
    for f in facts:
        key = (f.metric_canonical_id, f.period.year, f.period.quarter)
        if f.is_restated and key not in seen_restated:
            seen_restated.add(key)
            warnings.append(Warning(
                severity='info',
                message=f'{f.metric_canonical_id} {_format_period(f.period)} is a restated figure.',
            ))
 
    # Check 6: metric canonicalization ambiguity. Deduplicated per distinct
    # flag text, same reasoning as Checks 5/10: several corroborating
    # source tables sharing the same unresolved raw label would otherwise
    # each produce an identical warning.
    seen_ambiguity_flags = set()
    for f in facts:
        for flag in f.ambiguity_flags:
            if flag and flag not in seen_ambiguity_flags:
                seen_ambiguity_flags.add(flag)
                warnings.append(Warning(
                    severity='warning',
                    message=f'Metric label matched with ambiguity: {flag}',
                ))
        if f.ambiguity_flags:
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

    # Check 10: unaudited figures. Named explicitly in the assignment
    # brief's ambiguity checklist ("unaudited statements"). A 10-Q's
    # quarterly figures are unaudited by standard SEC practice; that is
    # normal, not a defect, so it is disclosed as info rather than
    # downgrading confidence -- the same treatment Check 5 gives restated
    # figures. Deduplicated per (metric, period) for the same reason.
    seen_unaudited = set()
    for f in facts:
        key = (f.metric_canonical_id, f.period.year, f.period.quarter)
        if not f.is_audited and key not in seen_unaudited:
            seen_unaudited.add(key)
            warnings.append(Warning(
                severity='info',
                message=f'{f.metric_canonical_id} {_format_period(f.period)} is from an unaudited (quarterly) filing.',
            ))

    log.info(
        "question_type=%s, %d facts -> confidence=%s (%d warnings)",
        plan.question_type, len(facts), confidence, len(warnings),
    )
    return warnings, confidence
