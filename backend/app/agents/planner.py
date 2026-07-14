""" The Planner uses an LLM with a strict Pydantic schema to convert natural language into a structured plan.
The schema is enforced by function-calling / structured-output modes, so the LLM cannot return free text
where a typed field is expected.

The API-enforced tool schema constrains field names and JSON types; it does not enforce the domain rules in
SYSTEM_PROMPT (e.g. which question_type to pick) or cross-field invariants, so the result is still validated
against QueryPlan with Pydantic afterward, and any residual failure still falls closed to UNKNOWN rather than
guessing.

metric_canonical_id is deliberately left for the Planner to leave null in normal use: the canonical metric
registry lives in ingest/canonicalizer.py, not here, and agents/fact_retriever.py already resolves
metric_natural_language against that registry deterministically. Teaching the Planner the full registry inline
would duplicate a registry that changes independently of this prompt.
"""
import logging

import anthropic
from app.schemas import QueryPlan, QuestionType, Period
from app.instrumentation import log_latency

log = logging.getLogger(__name__)
client = anthropic.Anthropic()

SYSTEM_PROMPT = '''You classify SEC filing questions into structured plans.
Rules:
1. If the question asks for a growth rate, delta, or year-over-year change in ONE metric that is a plain dollar figure or count (not a ratio or percentage), use question_type=growth_calc and include BOTH periods.
2. If the question asks for a single value at a point in time, use question_type=numeric_lookup.
3. If the question asks about management commentary, risks, or narrative content, use question_type=narrative.
4. Never invent a company or period. If unclear, add an ambiguity_flag.
5. Default gaap_preference to "gaap" unless the question says "non-GAAP" or "adjusted".
6. Extract the metric as plain text into metric_natural_language (e.g. "total revenue", "net income"). Do
   not invent a value for metric_canonical_id; leave it unset. A separate deterministic stage resolves the
   canonical metric ID from metric_natural_language.
7. A period's "quarter" field is the fiscal quarter number (1-4), not a calendar month or date. Leave it
   unset for a full fiscal year. Leave "is_ttm" and "is_ytd" false unless the question explicitly asks for a
   trailing-twelve-month or year-to-date figure.
8. If the question asks for a RATIO or PERCENTAGE relating TWO DIFFERENT metrics -- margin, "X as a percentage
   of Y", "X to Y ratio" -- use question_type=margin_calc, REGARDLESS of how many periods are involved. This
   includes questions phrased as a change or trend in that ratio, e.g. "how did gross margin change",
   "how much did SG&A grow as a percentage of revenue": these still name TWO metrics (numerator and
   denominator) needing a ratio, so they are margin_calc with TWO periods, never growth_calc. Put the metric
   before "as a percentage of" / the numerator (e.g. "gross profit" in "gross margin", "SG&A" in "SG&A as a
   percentage of revenue") in metric_natural_language, and the metric after "of" / the denominator (e.g.
   "revenue" in both examples) in ratio_denominator_natural_language. growth_calc is reserved for a single
   plain metric growing over time (rule 1); it never applies when the question names or implies two different
   metrics related by division.
9. If the question asks to RANK or compare MANY metrics against each other to find which changed the most --
   not one named metric across two periods, but "which metrics", "what were the biggest", "which line items" --
   use question_type=ranking with exactly TWO periods (the before/after periods) and leave
   metric_natural_language unset (ranking scans every metric, not one). Set ranking_direction: "top_increase"
   for "grew the most" / "biggest increases"; "top_decrease" for "declined the most" / "biggest decreases";
   "most_deteriorated" for "deteriorated most" / "got worse" / "worst performing"; "most_improved" for
   "improved most" / "best performing". Set ranking_scope="expense" only if the question explicitly restricts
   to expenses/costs (e.g. "expense increases", "cost line items"); otherwise leave it unset.
'''

PLAN_TOOL = {
    'name': 'submit_query_plan',
    'description': 'Submit the structured query plan extracted from the user question.',
    'input_schema': QueryPlan.model_json_schema(),
}

@log_latency(log)
def plan_query(question: str) -> QueryPlan:
    try:
        response = client.messages.create(
            model='claude-sonnet-4-5',
            max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=[PLAN_TOOL],
            tool_choice={'type': 'tool', 'name': 'submit_query_plan'},
            messages=[{'role': 'user', 'content': question}],
        )
        tool_use = next(b for b in response.content if b.type == 'tool_use')
        plan = QueryPlan(**tool_use.input)
    except Exception as e:
        # Fail closed: unknown plan instead of guessing
        plan = QueryPlan(
            question_type=QuestionType.UNKNOWN,
            ambiguity_flags=[f'plan_parse_failed: {e}'],
        )
    log.info(
        "question=%r -> question_type=%s, company=%s, metric=%r",
        question, plan.question_type.value, plan.company_ticker, plan.metric_natural_language,
    )
    return plan
