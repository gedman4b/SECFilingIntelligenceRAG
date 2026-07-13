"""
agents/prose_retriever.py

Prose Retriever Agent: vector search over narrative sections only.

Design principle
-----------------
Deterministic vector search. No LLM. Narrative questions ("What did
management cite as risks?") do not have a single ground-truth value, so
paraphrase-tolerant retrieval is the right tool here, exactly the case
embeddings are good at per Part 1 of the write-up. This stage only ever
calls store/vector_store.py, which is structurally incapable of embedding
a Fact or a table cell, so numeric lookup can never route through this
file, even by accident.

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
from typing import List

from app.schemas import QueryPlan
from app.store.vector_store import ProsePassage, query_prose
from app.instrumentation import log_latency

log = logging.getLogger(__name__)

DEFAULT_N_RESULTS = 5


@log_latency(log)
def retrieve_prose(plan: QueryPlan, question: str) -> List[ProsePassage]:
    """Retrieve narrative passages relevant to a natural-language question.

    Args:
        plan: Structured plan from the Query Planner. Only its
            company_ticker is used to scope the search; narrative
            questions carry no metric or period.
        question: The user's raw question text, embedded directly as the
            search query so paraphrase is preserved.

    Returns:
        Passages ordered by similarity, most relevant first. An empty
        list means no evidence was found and must be treated as such by
        the Verifier, not as "no risks/commentary exist in the filing".
    """
    passages = query_prose(
        query_text=question,
        n_results=DEFAULT_N_RESULTS,
        company_ticker=plan.company_ticker,
    )
    log.info(
        "Prose retrieval for %r (company=%s): %d passages",
        question, plan.company_ticker, len(passages),
    )
    return passages


# =============================================================================
# CLI entry point (manual testing against the persisted vector index)
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Run the Prose Retriever against the persisted vector index.",
    )
    ap.add_argument("question", help="Narrative question text")
    ap.add_argument("--company-ticker", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    plan = QueryPlan(question_type="narrative", company_ticker=args.company_ticker)
    found = retrieve_prose(plan, args.question)
    print(json.dumps([p.model_dump() for p in found], indent=2))
