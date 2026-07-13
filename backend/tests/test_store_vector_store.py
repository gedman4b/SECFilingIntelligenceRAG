"""Tests for store/vector_store.py.

Uses chromadb's bundled local embedding model (onnxruntime) -- not a live
LLM provider, so this stays in the default suite per AGENTS.md's rule
about network access to LLM providers specifically.
"""

from __future__ import annotations

from app.store.db import ProseChunkRecord
from app.store.vector_store import add_prose_chunks, query_prose


def test_query_prose_filters_by_company(chroma_collection):
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

    results = query_prose(
        "supply chain risk", n_results=5, company_ticker="TSLA",
        collection=chroma_collection,
    )
    assert len(results) == 1
    assert results[0].filing_id == "TSLA-10K-2025-12-31"


def test_query_prose_returns_provenance(chroma_collection):
    add_prose_chunks([ProseChunkRecord(
        id="tsla::risk::0", filing_id="TSLA-10K-2025-12-31", section_type="risk_factors",
        heading="Item 1A. Risk Factors", page_start=18, page_end=54, chunk_index=0,
        text="Supply chain disruption risk text.",
    )], company_ticker="TSLA", collection=chroma_collection)

    results = query_prose("supply chain", n_results=1, collection=chroma_collection)
    assert len(results) == 1
    passage = results[0]
    assert passage.page_start == 18
    assert passage.page_end == 54
    assert passage.section_type == "risk_factors"


def test_query_prose_empty_collection_returns_empty(chroma_collection):
    assert query_prose("anything", collection=chroma_collection) == []
