"""
All arithmetic lives here. No LLM. Behavior is fully deterministic and testable.
"""
from app.schemas import QueryPlan, Fact, QuestionType
from typing import List, Tuple, Optional
 
def compute(plan: QueryPlan, facts: List[Fact]) -> Tuple[Optional[float], Optional[str]]:
    if plan.question_type == QuestionType.GROWTH_CALC:
        if len(facts) < 2:
            return None, None  # let verifier catch the missing data
        # Sort by period so the older value is first
        sorted_facts = sorted(facts, key=lambda f: (f.period.year, f.period.quarter or 0))
        a = sorted_facts[0].value
        b = sorted_facts[-1].value
        if a == 0:
            return None, f'growth undefined: base period value is 0'
        growth_pct = ((b - a) / abs(a)) * 100
        expr = f'(({b:,} - {a:,}) / |{a:,}|) * 100 = {growth_pct:.2f}%'
        return round(growth_pct, 2), expr
 
    if plan.question_type == QuestionType.COMPARISON:
        # Simple delta rather than percentage
        if len(facts) < 2:
            return None, None
        sorted_facts = sorted(facts, key=lambda f: (f.period.year, f.period.quarter or 0))
        delta = sorted_facts[-1].value - sorted_facts[0].value
        expr = f'{sorted_facts[-1].value:,} - {sorted_facts[0].value:,} = {delta:,}'
        return delta, expr
 
    return None, None
