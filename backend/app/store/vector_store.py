"""
store/vector_store.py

Chroma vector index over prose only.

Design principle
-----------------
This is the retrieval mechanism the write-up calls the primary
architectural guard against near-miss label collisions (Guard 1: "numeric
queries never use embeddings"). It intentionally knows how to embed
exactly one thing: ProseChunkRecord objects from store/db.py. There is no
code path here that accepts a Fact, a table cell, or any other numeric
content. Numeric lookup stays entirely in the SQL fact store
(agents/fact_retriever.py); this module never sees a dollar value.

Embeddings are used exactly where the write-up says they work well:
paraphrase-tolerant retrieval over MD&A, risk factors, and other narrative
sections, where topical similarity, not exact label matching, is the
right retrieval signal.

Persistence and embedding model
--------------------------------
Uses chromadb's PersistentClient with its bundled default embedding
function (all-MiniLM-L6-v2, ONNX, runs locally via onnxruntime). No
additional embedding provider dependency is introduced; chromadb already
ships this model and its runtime.

Dependencies
------------
chromadb >= 1.5   (pulls in onnxruntime for the default embedding function)
pydantic >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel

from app.store.db import ProseChunkRecord

log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_data"
COLLECTION_NAME = "prose_chunks"


_vercel_persist_dir_ready = False


def _vercel_writable_persist_dir() -> str:
    """Return a writable copy of chroma_data for Vercel's deployed runtime.

    Vercel's deployed filesystem is read-only everywhere except /tmp.
    chromadb.PersistentClient needs to open (and, on first touch, write
    to) its own SQLite-backed tenant/database metadata store even for a
    read-only query -- confirmed via a live Vercel function traceback,
    not a guess: opening the bundled (read-only) chroma_data directly
    fails with a cryptic "Could not connect to tenant default_tenant. Are
    you sure it exists?" raised from deep inside chromadb's Rust
    bindings, which gives no hint that the real cause is a read-only
    filesystem. So on Vercel the bundled data is copied to /tmp once per
    cold start (memoized so repeat requests on a warm instance don't
    re-copy) and that writable copy is used instead.

    Returns:
        Path to the writable /tmp copy of chroma_data.
    """
    global _vercel_persist_dir_ready
    # chromadb's bundled ONNX embedding model also downloads to
    # $HOME/.cache on first use; $HOME may not be set to a writable
    # directory by default in Vercel's Python runtime.
    os.environ.setdefault("HOME", "/tmp")
    tmp_dir = "/tmp/chroma_data"
    if not _vercel_persist_dir_ready:
        if not os.path.exists(tmp_dir) and DEFAULT_PERSIST_DIR.exists():
            shutil.copytree(DEFAULT_PERSIST_DIR, tmp_dir)
        _vercel_persist_dir_ready = True
    return tmp_dir


def _resolve_persist_dir() -> str:
    """Return the Chroma persistence directory, honoring CHROMA_PERSIST_DIR.

    Returns:
        Absolute path to the Chroma persistence directory.
    """
    override = os.environ.get("CHROMA_PERSIST_DIR")
    if override:
        return override
    if os.environ.get("VERCEL"):
        return _vercel_writable_persist_dir()
    return str(DEFAULT_PERSIST_DIR)


def get_collection(persist_dir: Optional[str] = None) -> Collection:
    """Open (creating if needed) the prose-only Chroma collection.

    Args:
        persist_dir: Override path to the Chroma data directory. Defaults
            to the CHROMA_PERSIST_DIR environment variable, then
            DEFAULT_PERSIST_DIR.

    Returns:
        A Chroma Collection scoped to narrative prose chunks, using
        cosine distance.
    """
    client = chromadb.PersistentClient(path=persist_dir or _resolve_persist_dir())
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# Query result schema
# =============================================================================

class ProsePassage(BaseModel):
    """A retrieved prose passage with provenance, ready for citation."""
    chunk_id: str
    filing_id: str
    section_type: str
    heading: Optional[str] = None
    page_start: int
    page_end: int
    text: str
    distance: float


# =============================================================================
# Writes
# =============================================================================

def add_prose_chunks(
    chunks: List[ProseChunkRecord],
    company_ticker: str,
    collection: Optional[Collection] = None,
) -> None:
    """Embed and upsert prose chunks into the vector index.

    Args:
        chunks: Prose chunks to embed, as produced by the ingestion
            pipeline and already stored in the SQLite prose_chunks table.
        company_ticker: Ticker the chunks belong to, denormalized into
            Chroma metadata so queries can filter by company without a
            round trip to the fact store.
        collection: Reuse an existing collection handle; opens one via
            get_collection() if not provided.
    """
    if not chunks:
        return
    col = collection or get_collection()
    col.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "filing_id": c.filing_id,
                "company_ticker": company_ticker,
                "section_type": c.section_type,
                "heading": c.heading or "",
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )
    log.info("Embedded %d prose chunks for %s", len(chunks), company_ticker)


# =============================================================================
# Queries
# =============================================================================

def query_prose(
    query_text: str,
    n_results: int = 5,
    company_ticker: Optional[str] = None,
    filing_id: Optional[str] = None,
    section_type: Optional[str] = None,
    collection: Optional[Collection] = None,
) -> List[ProsePassage]:
    """Retrieve the most relevant prose passages for a narrative question.

    Numeric lookups must never call this function; retrieval for numeric
    facts is the deterministic SQL path in agents/fact_retriever.py.

    Args:
        query_text: Natural-language question or search text.
        n_results: Maximum number of passages to return.
        company_ticker: Optional filter to a single company.
        filing_id: Optional filter to a single filing.
        section_type: Optional filter to a section type ('mdna',
            'risk_factors', 'business', ...).
        collection: Reuse an existing collection handle; opens one via
            get_collection() if not provided.

    Returns:
        Passages ordered by similarity, most relevant first.
    """
    col = collection or get_collection()

    conditions = []
    if company_ticker:
        conditions.append({"company_ticker": {"$eq": company_ticker}})
    if filing_id:
        conditions.append({"filing_id": {"$eq": filing_id}})
    if section_type:
        conditions.append({"section_type": {"$eq": section_type}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    result = col.query(query_texts=[query_text], n_results=n_results, where=where)

    passages: List[ProsePassage] = []
    ids = result.get("ids") or [[]]
    docs = result.get("documents") or [[]]
    metas = result.get("metadatas") or [[]]
    dists = result.get("distances") or [[]]
    for chunk_id, text, meta, dist in zip(ids[0], docs[0], metas[0], dists[0]):
        passages.append(ProsePassage(
            chunk_id=chunk_id,
            filing_id=meta["filing_id"],
            section_type=meta["section_type"],
            heading=meta.get("heading") or None,
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            text=text,
            distance=dist,
        ))
    return passages


# =============================================================================
# CLI entry point (manual testing / one-off querying)
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Query the prose vector index manually.",
    )
    ap.add_argument("query", help="Natural-language search text")
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--company-ticker", default=None)
    ap.add_argument("--section-type", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    found = query_prose(
        args.query,
        n_results=args.n_results,
        company_ticker=args.company_ticker,
        section_type=args.section_type,
    )
    print(json.dumps([p.model_dump() for p in found], indent=2))
