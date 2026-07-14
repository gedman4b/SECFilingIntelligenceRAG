"""
All arithmetic lives here. No LLM. Behavior is fully deterministic and testable.
"""
import logging

from app.schemas import QueryPlan, Fact, QuestionType
from app.ingest.canonicalizer import expense_metric_ids
from app.instrumentation import log_latency
from typing import Dict, List, Tuple, Optional

log = logging.getLogger(__name__)

# How many ranked metrics to surface. Not user-configurable ("top 3
# expense increases" is not parsed from the question) -- a fixed cap keeps
# the answer readable and keeps this a deterministic, testable constant
# rather than free-text-derived behavior.
RANKING_TOP_N = 5

@log_latency(log)
def compute(
    plan: QueryPlan,
    facts: List[Fact],
    denominator_facts: Optional[List[Fact]] = None,
    metric_facts_by_id: Optional[Dict[str, List[Fact]]] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute the deterministic result for plan.question_type.

    Args:
        plan: The structured plan.
        facts: For growth_calc/comparison, the one metric's facts across
            periods. For margin_calc, the numerator metric's facts.
        denominator_facts: Only used for margin_calc: the denominator
            metric's facts (e.g. revenue, for a gross margin question).
            Ignored for every other question_type.
        metric_facts_by_id: Only used for ranking: every candidate
            metric's facts across the two requested periods, from
            agents/fact_retriever.retrieve_ranking_facts(). Ignored for
            every other question_type.

    Returns:
        (computed_value, computation_expression), or (None, None) /
        (None, "<reason>") if there isn't enough data to compute.
    """
    result: Tuple[Optional[float], Optional[str]] = (None, None)

    if plan.question_type == QuestionType.MARGIN_CALC:
        if not facts or not denominator_facts:
            result = (None, None)  # let verifier catch the missing data
        else:
            # A margin is a ratio within ONE period, not across periods.
            # A "last two years" question needs a ratio for EACH of those
            # periods, not just the latest: live testing showed that
            # giving the Composer only one pre-computed ratio plus raw
            # facts for a second period left it "helpfully" computing that
            # second ratio itself, which is exactly the arithmetic the
            # Composer must never do. So every common period gets computed
            # here and folded into one expression string; computed_value
            # stays the single most recent figure (matching every other
            # question_type's one-number contract), but the full,
            # ready-to-quote breakdown is in computation_expression, so the
            # Composer has nothing left to compute.
            numerator_by_period = {(f.period.year, f.period.quarter): f for f in facts}
            denominator_by_period = {(f.period.year, f.period.quarter): f for f in denominator_facts}
            common_periods = sorted(
                set(numerator_by_period) & set(denominator_by_period), reverse=True,
            )
            if not common_periods:
                result = (None, 'no period has both the numerator and denominator metric')
            else:
                lines: List[str] = []
                primary_value: Optional[float] = None
                computed_margins: List[Tuple[str, float]] = []  # (label, margin_pct), most-recent first
                for period_key in common_periods:
                    year, quarter = period_key
                    num = numerator_by_period[period_key]
                    den = denominator_by_period[period_key]
                    label = f'Y{year}Q{quarter or "FY"}'
                    if den.value == 0:
                        lines.append(f'{label}: margin undefined (denominator is 0)')
                        continue
                    margin_pct = round((num.value / den.value) * 100, 2)
                    lines.append(f'{label}: ({num.value:,} / {den.value:,}) * 100 = {margin_pct:.2f}%')
                    computed_margins.append((label, margin_pct))
                    if primary_value is None:
                        primary_value = margin_pct
                if primary_value is None:
                    result = (None, 'margin undefined for every requested period: denominator value is 0')
                else:
                    # A "how much did X grow as a % of Y" question implies a
                    # delta between two already-computed ratios. Live testing
                    # showed that leaving this one subtraction to the
                    # Composer meant it did the arithmetic itself ("an
                    # increase of 0.88 percentage points") even though every
                    # input was already given -- still a forbidden
                    # computation. So the delta between the most recent and
                    # oldest computed margin is folded in here too, leaving
                    # nothing left to compute.
                    if len(computed_margins) >= 2:
                        newest_label, newest_pct = computed_margins[0]
                        oldest_label, oldest_pct = computed_margins[-1]
                        delta_pp = round(newest_pct - oldest_pct, 2)
                        lines.append(
                            f'Change from {oldest_label} to {newest_label}: '
                            f'{newest_pct:.2f}% - {oldest_pct:.2f}% = {delta_pp:+.2f} percentage points'
                        )
                    result = (primary_value, '; '.join(lines))

    elif plan.question_type == QuestionType.GROWTH_CALC:
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

    elif plan.question_type == QuestionType.RANKING:
        if not metric_facts_by_id:
            result = (None, None)  # let verifier catch the missing data
        else:
            expense_ids = set(expense_metric_ids())
            candidates = []  # (metric_id, label, older_value, newer_value, growth_pct)
            for metric_id, mfacts in metric_facts_by_id.items():
                sorted_mfacts = sorted(mfacts, key=lambda f: (f.period.year, f.period.quarter or 0))
                older, newer = sorted_mfacts[0], sorted_mfacts[-1]
                if older.value == 0:
                    continue  # undefined growth for this one metric; skip it, don't fail the whole ranking
                growth_pct = round(((newer.value - older.value) / abs(older.value)) * 100, 2)
                candidates.append((metric_id, older.metric_raw_label, older.value, newer.value, growth_pct))

            if not candidates:
                result = (None, 'no candidate metric had a usable (non-zero base) value in both requested periods')
            else:
                direction = plan.ranking_direction
                if direction == 'top_increase':
                    candidates.sort(key=lambda c: c[4], reverse=True)
                elif direction == 'top_decrease':
                    candidates.sort(key=lambda c: c[4])
                else:
                    # most_deteriorated / most_improved: a rising expense
                    # and a falling revenue are both deterioration, so sort
                    # by "badness" (positive for a worsening expense,
                    # positive for a shrinking revenue/profit metric)
                    # rather than raw growth_pct, which would conflate the
                    # two.
                    def badness(c):
                        metric_id, _label, _a, _b, pct = c
                        return pct if metric_id in expense_ids else -pct
                    candidates.sort(key=badness, reverse=(direction == 'most_deteriorated'))

                top = candidates[:RANKING_TOP_N]
                lines = [
                    f'{i}. {label} ({metric_id}): {a:,} -> {b:,} ({pct:+.2f}%)'
                    for i, (metric_id, label, a, b, pct) in enumerate(top, start=1)
                ]
                result = (top[0][4], '; '.join(lines))

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
