"""
All arithmetic lives here. No LLM. Behavior is fully deterministic and testable.
"""
import logging

from app.schemas import QueryPlan, Fact, QuestionType
from app.instrumentation import log_latency
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

@log_latency(log)
def compute(plan: QueryPlan, facts: List[Fact]) -> Tuple[Optional[float], Optional[str]]:
    result: Tuple[Optional[float], Optional[str]] = (None, None)

    if plan.question_type == QuestionType.GROWTH_CALC:
        if len(facts) < 2:
            result = (None, None)  # let verifier catch the missing data
        else:
            # Sort by period so the older value is first
            sorted_facts = sorted(facts, key=lambda f: (f.period.year, f.period.quarter or 0))
            a = sorted_facts[0].value
            b = sorted_facts[-1].value
            if a == 0:
                result = (None, f'growth undefined: base period value is 0')
            else:
                growth_pct = ((b - a) / abs(a)) * 100
                expr = f'(({b:,} - {a:,}) / |{a:,}|) * 100 = {growth_pct:.2f}%'
                result = (round(growth_pct, 2), expr)

    elif plan.question_type == QuestionType.COMPARISON:
        # Simple delta rather than percentage
        if len(facts) < 2:
            result = (None, None)
        else:
            sorted_facts = sorted(facts, key=lambda f: (f.period.year, f.period.quarter or 0))
            delta = sorted_facts[-1].value - sorted_facts[0].value
            expr = f'{sorted_facts[-1].value:,} - {sorted_facts[0].value:,} = {delta:,}'
            result = (delta, expr)

    log.info("question_type=%s, %d facts -> computed=%s", plan.question_type.value, len(facts), result[0])
    return result
