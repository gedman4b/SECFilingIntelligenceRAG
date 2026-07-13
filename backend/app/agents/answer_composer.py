"""
The Composer uses an LLM to narrate the response, but is given only pre-verified facts and computed values. 
It cannot introduce arithmetic error or hallucinate values because it does not have access to filing content or a calculator.
"""
import logging

import anthropic
from app.schemas import QueryResponse, Fact, Warning, QueryPlan
from app.store.vector_store import ProsePassage
from app.instrumentation import log_latency
from typing import List, Optional

log = logging.getLogger(__name__)
client = anthropic.Anthropic()

COMPOSE_SYSTEM = '''You are composing an answer to a financial question.
You are given: the user's question, a structured plan, retrieved facts
with provenance, an optional computed value, narrative passages with
provenance, and warnings.

You MUST:
1. Only use the facts, computed values, and narrative passages provided. Do not introduce any numbers or claims that are not grounded in them.
2. Never perform arithmetic yourself. Do not add, subtract, multiply, divide, or otherwise combine any of the provided fact values into a new number, even if the result looks obvious. State a calculated result (a delta, growth rate, margin, or ratio) ONLY if it is given to you verbatim as "Computed value" / "Calculation" below. If no computed value is given, cite the individual raw facts side by side and let the reader compare them; do not state what their difference or ratio is.
3. State the answer in a single sentence, then show the calculation if applicable, quoting the given Calculation expression verbatim rather than restating it in your own words.
4. For narrative questions, synthesize the answer only from the provided passages, and cite each claim inline with its section and page, e.g. "(Risk Factors, page 12)".
5. Surface every warning verbatim in a Warnings section.
6. Cite each fact by filing URL, page, and table.
7. If confidence is insufficient_data, say so explicitly.
8. Never claim precision the source does not have.
'''

@log_latency(log)
def compose_answer(
    question: str,
    plan: QueryPlan,
    facts: List[Fact],
    computed: Optional[float],
    comp_expr: Optional[str],
    prose: List[ProsePassage],
    warnings: List[Warning],
    confidence: str,
) -> QueryResponse:
    if confidence == 'insufficient_data':
        answer_text = ('I cannot answer this question from the available filings. ' +
                       'Reason(s): ' + '; '.join(w.message for w in warnings
                       if w.severity == 'error'))
        log.info("confidence=insufficient_data -> composed without an LLM call")
        return QueryResponse(
            answer_text=answer_text,
            raw_facts=facts,
            warnings=warnings,
            citations=[],
            confidence=confidence,
        )
 
    facts_summary = '\n'.join([
        f'- {f.metric_raw_label} for {f.period.year} Q{f.period.quarter or "FY"}: ' +
        f'{f.value:,} {f.units}, GAAP={f.is_gaap}, page {f.page_number}'
        for f in facts
    ])

    prose_summary = '\n'.join([
        f'- [{p.section_type}, page {p.page_start}] {p.text}'
        for p in prose
    ])

    user_msg = (f'Question: {question}\n\n' +
                f'Facts:\n{facts_summary}\n\n' +
                (f'Narrative passages:\n{prose_summary}\n\n' if prose else '') +
                (f'Computed value: {computed}\n' if computed is not None else '') +
                (f'Calculation: {comp_expr}\n' if comp_expr else '') +
                f'Warnings: {[w.message for w in warnings]}\n' +
                f'Confidence: {confidence}')
 
    response = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=600,
        system=COMPOSE_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
 
    citations = [{
        'filing_url': f.filing_url,
        'page': f.page_number,
        'table_id': f.table_id,
        'label': f.metric_raw_label,
        'value': f.value,
        'units': f.units,
        'period': f'{f.period.year} Q{f.period.quarter}' if f.period.quarter else str(f.period.year),
    } for f in facts]
 
    log.info(
        "confidence=%s, %d facts, %d prose passages -> %d citations",
        confidence, len(facts), len(prose), len(citations),
    )
    return QueryResponse(
        answer_text=response.content[0].text,
        raw_facts=facts,
        computed_value=computed,
        computation_expression=comp_expr,
        warnings=warnings,
        citations=citations,
        confidence=confidence,
    )
