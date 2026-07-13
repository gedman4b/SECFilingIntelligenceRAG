"""Tests for agents/prose_retriever.py. Deterministic vector search, no LLM
call -- uses chromadb's local embedding model, same as store/vector_store.py."""

from __future__ import annotations

from app.agents.prose_retriever import retrieve_prose
from app.schemas import QueryPlan, QuestionType
from app.store.db import ProseChunkRecord
from app.store.vector_store import add_prose_chunks


def test_retrieve_prose_scoped_to_company(chroma_collection):
    add_prose_chunks([ProseChunkRecord(
        id="tsla::risk::0", filing_id="TSLA-10K-2025-12-31", section_type="risk_factors",
        heading="Item 1A", page_start=18, page_end=54, chunk_index=0,
        text="We rely on a limited number of suppliers for battery cells.",
    )], company_ticker="TSLA", collection=chroma_collection)
    add_prose_chunks([ProseChunkRecord(
        id="aapl::risk::0", filing_id="AAPL-10K-2025-09-27", section_type="risk_factors",
        heading="Item 1A", page_start=8, page_end=30, chunk_index=0,
        text="The Company depends on component suppliers outside the United States.",
    )], company_ticker="AAPL", collection=chroma_collection)

    plan = QueryPlan(question_type=QuestionType.NARRATIVE, company_ticker="TSLA")
    passages = retrieve_prose(plan, "What supply chain risks does the company face?")
    assert len(passages) == 1
    assert passages[0].filing_id == "TSLA-10K-2025-12-31"


def test_retrieve_prose_no_company_filter_returns_all(chroma_collection):
    add_prose_chunks([ProseChunkRecord(
        id="tsla::risk::0", filing_id="TSLA-10K-2025-12-31", section_type="risk_factors",
        heading="Item 1A", page_start=18, page_end=54, chunk_index=0,
        text="Supply chain risk from battery suppliers.",
    )], company_ticker="TSLA", collection=chroma_collection)
    add_prose_chunks([ProseChunkRecord(
        id="aapl::risk::0", filing_id="AAPL-10K-2025-09-27", section_type="risk_factors",
        heading="Item 1A", page_start=8, page_end=30, chunk_index=0,
        text="Supply chain risk from component suppliers.",
    )], company_ticker="AAPL", collection=chroma_collection)

    plan = QueryPlan(question_type=QuestionType.NARRATIVE)
    passages = retrieve_prose(plan, "supply chain risk")
    assert len(passages) == 2


def test_retrieve_prose_empty_index_returns_empty(chroma_collection):
    plan = QueryPlan(question_type=QuestionType.NARRATIVE, company_ticker="TSLA")
    assert retrieve_prose(plan, "anything") == []
