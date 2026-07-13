"""
ingest/run_ingestion.py

Ingestion orchestration: runs the full offline pipeline (pdf_parser ->
table_classifier -> fact_extractor -> canonicalizer, plus prose chunking)
over every filing in ingest/pdfs/ and populates the persistent fact store
and vector index.

Design principle
-----------------
This is the "offline pipeline" from Part 2 of the write-up, run once per
filing. It is intentionally the only place in the codebase that makes LLM
calls at ingestion scale (table_classifier and fact_extractor, once per
table), and it is expected to be slow: a single large statement table can
take a minute and several thousand output tokens (see
EXTRACTION_MAX_TOKENS in fact_extractor.py). That cost is deliberate --
it is what lets the online query path stay LLM-free and fast for numeric
questions (Part 1: "The offline pipeline pays the interpretation cost.
The online pipeline benefits from the structure."). This script is not
part of the query pipeline eval/runner.py exercises and main.py never
calls it; it is a standalone, manually-triggered job, matching Part 2's
framing of ingestion as "per filing, one-time."

Per-filing metadata below (company, form type, fiscal year/quarter,
period end date) was read directly off each filing's actual cover page,
not inferred from filename: a filename's date and the period a filing
covers can disagree. Concretely, tsla-10-K-A 20260430.pdf is dated by its
amendment filing date, but its cover page states it covers the fiscal
year ended December 31, 2025 -- the same period as the original 10-K.
Both are ingested as separate filing_ids covering the same period; this
script does not attempt to resolve which one supersedes the other for
retrieval. Part 9 of the write-up names this as a known, unaddressed
prototype weakness: "When a 10-K/A supersedes a 10-K, the current system
stores both and returns whichever matches first."

Prose chunking granularity: pdf_parser.py's ProseSection spans an entire
filing section (Tesla's risk_factors section alone spans pages 18-54), far
too large to embed as one unit. This module greedily packs the section's
text into ~1500-character chunks for embedding. Every chunk from a given
section inherits that section's overall page_start/page_end, not a
tighter per-chunk page range, because pdf_parser.py's section extraction
does not track which physical page each line of joined text came from.
Prose citations from this pipeline are therefore accurate to a page
range, not a single page -- consistent with what the parser actually
knows, not overstated precision.

Dependencies
------------
anthropic, pydantic >= 2.0 (via the ingest/ and store/ modules this
orchestrates)

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.ingest.canonicalizer import canonicalize_facts
from app.ingest.fact_extractor import FilingContext, extract_facts_from_tables
from app.ingest.pdf_parser import ProseSection, parse_filing
from app.ingest.table_classifier import classify_tables
from app.store.db import (
    FilingRecord,
    ProseChunkRecord,
    get_conn,
    insert_fact,
    insert_filing,
    insert_prose_chunk,
)
from app.store.vector_store import add_prose_chunks, get_collection

log = logging.getLogger(__name__)

PDF_DIR = Path(__file__).resolve().parent / "pdfs"

# Roughly 300-400 tokens: large enough to preserve paragraph context,
# small enough that a single embedded chunk stays topically coherent.
CHUNK_TARGET_CHARS = 1500


@dataclass(frozen=True)
class FilingSpec:
    """Metadata for one filing to ingest, read from its actual cover page."""
    pdf_filename: str
    filing_id: str
    company_ticker: str
    form_type: str
    fiscal_year: int
    fiscal_quarter: Optional[int]
    period_end_date: str
    is_amendment: bool = False


FILINGS: List[FilingSpec] = [
    FilingSpec(
        pdf_filename="aapl-10-K 20250927.pdf",
        filing_id="AAPL-10K-2025-09-27",
        company_ticker="AAPL", form_type="10-K",
        fiscal_year=2025, fiscal_quarter=None, period_end_date="2025-09-27",
    ),
    FilingSpec(
        pdf_filename="aapl-10-Q 20251227.pdf",
        filing_id="AAPL-10Q-2025-12-27",
        company_ticker="AAPL", form_type="10-Q",
        fiscal_year=2026, fiscal_quarter=1, period_end_date="2025-12-27",
    ),
    FilingSpec(
        pdf_filename="aapl-10-Q 20260328.pdf",
        filing_id="AAPL-10Q-2026-03-28",
        company_ticker="AAPL", form_type="10-Q",
        fiscal_year=2026, fiscal_quarter=2, period_end_date="2026-03-28",
    ),
    FilingSpec(
        pdf_filename="tsla-10-K 20251231.pdf",
        filing_id="TSLA-10K-2025-12-31",
        company_ticker="TSLA", form_type="10-K",
        fiscal_year=2025, fiscal_quarter=None, period_end_date="2025-12-31",
    ),
    FilingSpec(
        pdf_filename="tsla-10-K-A 20260430.pdf",
        filing_id="TSLA-10KA-2025-12-31",
        company_ticker="TSLA", form_type="10-K/A",
        fiscal_year=2025, fiscal_quarter=None, period_end_date="2025-12-31",
        is_amendment=True,
    ),
    FilingSpec(
        pdf_filename="tsla-10-Q 20260331.pdf",
        filing_id="TSLA-10Q-2026-03-31",
        company_ticker="TSLA", form_type="10-Q",
        fiscal_year=2026, fiscal_quarter=1, period_end_date="2026-03-31",
    ),
]


# =============================================================================
# Per-filing ingestion
# =============================================================================

def ingest_filing(spec: FilingSpec, conn, chroma_collection) -> None:
    """Run the full offline pipeline for one filing and store the results.

    Args:
        spec: Metadata for the filing to ingest.
        conn: Open connection from store.db.get_conn().
        chroma_collection: Open collection from store.vector_store.get_collection().
    """
    pdf_path = PDF_DIR / spec.pdf_filename
    filing_url = f"file://{pdf_path}"

    log.info("Ingesting %s (%s)", spec.filing_id, spec.pdf_filename)
    parsed = parse_filing(str(pdf_path), spec.filing_id)
    for warning in parsed.warnings:
        log.warning("%s: parser warning: %s", spec.filing_id, warning)

    insert_filing(conn, FilingRecord(
        id=spec.filing_id, company_ticker=spec.company_ticker,
        form_type=spec.form_type, fiscal_year=spec.fiscal_year,
        fiscal_quarter=spec.fiscal_quarter, period_end_date=spec.period_end_date,
        filing_url=filing_url, filing_path=str(pdf_path),
        is_amendment=spec.is_amendment,
    ))

    classifications = classify_tables(parsed.tables)
    filing_context = FilingContext(
        form_type=spec.form_type, fiscal_year=spec.fiscal_year,
        fiscal_quarter=spec.fiscal_quarter,
    )
    extracted = extract_facts_from_tables(parsed.tables, classifications, filing_context)
    canonicalized = canonicalize_facts(extracted, spec.company_ticker)
    for record in canonicalized:
        insert_fact(conn, record)
    resolved = sum(1 for r in canonicalized if r.metric_canonical_id != "UNRESOLVED")
    log.info(
        "%s: %d tables classified, %d facts extracted (%d resolved to canonical metrics)",
        spec.filing_id, len(parsed.tables), len(canonicalized), resolved,
    )

    chunks: List[ProseChunkRecord] = []
    for section in parsed.prose_sections:
        chunks.extend(_chunk_prose_section(section))
    for chunk in chunks:
        insert_prose_chunk(conn, chunk)
    if chunks:
        add_prose_chunks(chunks, company_ticker=spec.company_ticker, collection=chroma_collection)
    log.info(
        "%s: %d prose sections chunked into %d embeddings",
        spec.filing_id, len(parsed.prose_sections), len(chunks),
    )


def _chunk_prose_section(section: ProseSection) -> List[ProseChunkRecord]:
    """Split one prose section's text into embedding-sized chunks.

    Greedy paragraph packing: splits on blank lines first; pdf_parser.py
    joins per-page text with single newlines rather than blank-line
    paragraph breaks, so this falls back to newline splitting when blank
    lines aren't found. Packs consecutive lines into a chunk up to
    CHUNK_TARGET_CHARS without splitting a line, so the only way a chunk
    exceeds the target is a single unusually long line.

    Args:
        section: A prose section extracted by ingest/pdf_parser.py.

    Returns:
        One or more chunks, in order, ready for store.db.insert_prose_chunk
        and store.vector_store.add_prose_chunks.
    """
    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in section.text.split("\n") if p.strip()]

    chunks: List[ProseChunkRecord] = []
    buffer: List[str] = []
    buffer_len = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, chunk_index
        if not buffer:
            return
        chunks.append(ProseChunkRecord(
            id=f"{section.section_id}::chunk{chunk_index}",
            filing_id=section.filing_id,
            section_type=section.section_type,
            heading=section.heading,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_index=chunk_index,
            text=" ".join(buffer),
        ))
        chunk_index += 1
        buffer = []
        buffer_len = 0

    for line in paragraphs:
        if buffer and buffer_len + len(line) > CHUNK_TARGET_CHARS:
            flush()
        buffer.append(line)
        buffer_len += len(line)
    flush()

    return chunks


# =============================================================================
# Batch orchestration
# =============================================================================

def run_ingestion(
    filings: Optional[List[FilingSpec]] = None,
    db_path: Optional[str] = None,
    chroma_dir: Optional[str] = None,
) -> List[str]:
    """Ingest every given filing into the fact store and vector index.

    A failure on one filing is logged and skipped rather than aborting the
    batch, matching pdf_parser.py's never-abort-on-one-failure pattern.

    Args:
        filings: Filings to ingest. Defaults to FILINGS (all 6 filings in
            ingest/pdfs/).
        db_path: Override the SQLite fact store path.
        chroma_dir: Override the Chroma persistence directory.

    Returns:
        filing_ids that failed to ingest. Empty if every filing succeeded.
    """
    filings = FILINGS if filings is None else filings
    if chroma_dir is not None:
        os.environ["CHROMA_PERSIST_DIR"] = chroma_dir

    conn = get_conn(db_path)
    collection = get_collection()

    failed: List[str] = []
    for spec in filings:
        try:
            ingest_filing(spec, conn, collection)
        except Exception as exc:
            log.error("%s: ingestion failed, skipping: %s", spec.filing_id, exc)
            failed.append(spec.filing_id)
    conn.close()

    log.info(
        "Ingestion run complete: %d/%d filings succeeded",
        len(filings) - len(failed), len(filings),
    )
    return failed


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Run the offline ingestion pipeline over every filing in ingest/pdfs/.",
    )
    ap.add_argument("--db-path", default=None, help="Override the SQLite fact store path")
    ap.add_argument("--chroma-dir", default=None, help="Override the Chroma persistence directory")
    ap.add_argument(
        "--only", default=None,
        help="Comma-separated filing_ids to ingest instead of all 6, e.g. TSLA-10K-2025-12-31",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    selected = FILINGS
    if args.only:
        wanted = set(args.only.split(","))
        selected = [f for f in FILINGS if f.filing_id in wanted]
        missing = wanted - {f.filing_id for f in selected}
        if missing:
            print(f"Unknown filing_id(s): {missing}")
            sys.exit(2)

    failures = run_ingestion(filings=selected, db_path=args.db_path, chroma_dir=args.chroma_dir)
    if failures:
        print(f"FAILED to ingest: {failures}")
        sys.exit(1)
    print(f"Ingested {len(selected)} filing(s) successfully.")
