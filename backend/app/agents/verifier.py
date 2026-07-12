"""
The Verifier is the trust boundary. Every check here corresponds to a specific failure mode from the previous system.
"""
from app.schemas import QueryPlan, Fact, Warning
from app.store.vector_store import ProsePassage
from typing import List, Tuple, Optional

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
 
    # Check 7: sanity band on computed growth rates
    if computed is not None and abs(computed) > 5000:
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
            return warnings, 'insufficient_data'

    return warnings, confidence
