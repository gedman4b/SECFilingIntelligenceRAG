"""
The Composer uses an LLM to narrate the response, but is given only pre-verified facts and computed values. 
It cannot introduce arithmetic error or hallucinate values because it does not have access to filing content or a calculator.
"""
import anthropic
from app.schemas import QueryResponse, Fact, Warning, QueryPlan
from app.store.vector_store import ProsePassage
from typing import List, Optional

client = anthropic.Anthropic()

COMPOSE_SYSTEM = '''You are composing an answer to a financial question.
You are given: the user's question, a structured plan, retrieved facts
with provenance, an optional computed value, narrative passages with
provenance, and warnings.

You MUST:
1. Only use the facts, computed values, and narrative passages provided. Do not introduce any numbers or claims that are not grounded in them.
2. State the answer in a single sentence, then show the calculation if applicable.
3. For narrative questions, synthesize the answer only from the provided passages, and cite each claim inline with its section and page, e.g. "(Risk Factors, page 12)".
4. Surface every warning verbatim in a Warnings section.
5. Cite each fact by filing URL, page, and table.
6. If confidence is insufficient_data, say so explicitly.
7. Never claim precision the source does not have.
'''

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
 
    return QueryResponse(
        answer_text=response.content[0].text,
        raw_facts=facts,
        computed_value=computed,
        computation_expression=comp_expr,
        warnings=warnings,
        citations=citations,
        confidence=confidence,
    )
