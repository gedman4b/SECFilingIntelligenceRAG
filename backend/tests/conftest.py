"""
tests/conftest.py

Shared pytest fixtures for the backend test suite.

Design principle
-----------------
AGENTS.md: "No test that requires network access to a live LLM provider
runs in the default suite. Mock at the client boundary." Every fixture
here that touches an LLM-calling module (planner, answer_composer,
table_classifier, fact_extractor) mocks the Anthropic client's
messages.create call rather than hitting the real API. store.vector_store
is the one exception that makes a real call: it uses chromadb's bundled
local embedding model (onnxruntime), not a live LLM provider -- no
network round trip happens once the model is cached locally.

Every database/vector-store fixture uses a temporary path per test
(tmp_path, a built-in pytest fixture), never the real persistent store at
store/fact_store.db, so tests never depend on or mutate ingested data.
"""

from __future__ import annotations

import os
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from app.ingest.pdf_parser import ParsedTable, TableCell
from app.store.db import get_conn
from app.store.vector_store import get_collection


@pytest.fixture
def sample_income_stmt_table() -> ParsedTable:
    """A small, real-shaped income statement table (values from Tesla's
    actual FY2025 10-K, page 61)."""
    return ParsedTable(
        filing_id="TSLA-10K-2025-12-31",
        table_id="TSLA-10K-2025-12-31::page61::table1",
        page_number=61,
        bbox=(0.0, 0.0, 500.0, 300.0),
        n_rows=3,
        n_cols=3,
        cells=[
            TableCell(row=0, col=0, value="Metric", is_header=True),
            TableCell(row=0, col=1, value="2025", is_header=True),
            TableCell(row=0, col=2, value="2024", is_header=True),
            TableCell(row=1, col=0, value="Total revenues"),
            TableCell(row=1, col=1, value="94,827"),
            TableCell(row=1, col=2, value="97,690"),
            TableCell(row=2, col=0, value="Net income"),
            TableCell(row=2, col=1, value="3,855"),
            TableCell(row=2, col=2, value="7,153"),
        ],
        caption="CONSOLIDATED STATEMENTS OF OPERATIONS",
    )


@pytest.fixture
def db_conn(tmp_path):
    """A fresh SQLite fact store at a temporary path, schema auto-created."""
    conn = get_conn(str(tmp_path / "test_fact_store.db"))
    yield conn
    conn.close()


@pytest.fixture
def chroma_collection(tmp_path, monkeypatch):
    """A fresh Chroma collection at a temporary directory."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_data"))
    return get_collection()


def make_mock_anthropic_response(text: str) -> MagicMock:
    """Build a fake anthropic Message response with one text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def make_mock_tool_use_response(tool_name: str, tool_input: dict) -> MagicMock:
    """Build a fake anthropic Message response with one tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"
    return response
