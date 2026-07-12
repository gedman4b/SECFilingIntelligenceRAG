"""
ingest/fact_extractor.py

Fact extraction: LLM extraction plus rule-based validation, run once per
classified table at ingestion time.

Design principle
-----------------
Two-pass extraction, matching Part 3 of the write-up. Pass one is the LLM:
it reads the table's column headers and row labels to identify each
(metric, period, value) fact, because that requires genuine language
understanding (what does "Total revenues" mean, which column is FY2025
versus FY2024). Pass two is deterministic Python: it re-validates every
value the LLM claims to have extracted rather than trusting it blindly.
A row that fails validation is dropped, not coerced, so a bad extraction
produces a missing fact (caught later as insufficient_data) rather than a
wrong one silently entering the fact store.

This stage does not canonicalize metric labels or assign a company
ticker; ingest/canonicalizer.py owns that. Facts leaving this module still
carry the raw label exactly as it appeared in the filing.

Tables classified as TableCategory.OTHER are skipped entirely: they are
not one of the financial statement categories, so there is nothing to
extract.

Dependencies
------------
anthropic  (LLM call)
pydantic >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import anthropic
from pydantic import BaseModel, ValidationError, field_validator

from app.ingest.pdf_parser import ParsedTable
from app.ingest.table_classifier import TableCategory, TableClassification, render_table_as_text

log = logging.getLogger(__name__)

client = anthropic.Anthropic()

MIN_PLAUSIBLE_YEAR = 1990
MAX_PLAUSIBLE_YEAR = 2035

# Large statement tables (e.g. a full income statement with per-share and
# share-count breakdowns) can produce 90+ facts; each fact costs roughly
# 90-100 output tokens in this schema. 16000 was verified against a 37-row,
# 9-column real table (92 facts, 8792 tokens used) with headroom to spare.
EXTRACTION_MAX_TOKENS = 16000


# =============================================================================
# Output schema
# =============================================================================

class ExtractedFact(BaseModel):
    """A single numeric fact extracted from a table, with full provenance,
    prior to metric canonicalization."""
    metric_raw_label: str
    year: int
    quarter: Optional[int] = None  # None = full year / period-to-date
    is_ttm: bool = False
    value: float
    units: str
    is_gaap: bool = True
    is_restated: bool = False
    filing_id: str
    page_number: int
    table_id: str
    row_id: int


# =============================================================================
# LLM output schema (pass one) -- validated independently of ExtractedFact
# so a malformed LLM row never silently becomes a "valid" fact.
# =============================================================================

class _RawFactRow(BaseModel):
    row_id: int
    metric_raw_label: str
    year: int
    quarter: Optional[int] = None
    is_ttm: bool = False
    value: float
    units: str
    is_gaap: bool = True
    is_restated: bool = False

    @field_validator("quarter")
    @classmethod
    def _quarter_in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (1, 2, 3, 4):
            raise ValueError(f"quarter must be 1-4 or null, got {v}")
        return v

    @field_validator("metric_raw_label")
    @classmethod
    def _label_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("metric_raw_label must not be blank")
        return v

    @field_validator("units")
    @classmethod
    def _units_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("units must not be blank")
        return v


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = '''You extract numeric facts from a single financial table
that has already been classified into a statement category.

For EVERY data row that names a distinct financial metric with a numeric
value, extract one fact per (metric, period) combination present in that
row. Skip subtotal underlines, blank rows, and rows that are pure section
headers with no value.

Return a JSON object: {"facts": [...]}, where each item has exactly these
fields:
{
  "row_id": the 0-indexed row number in the table as given,
  "metric_raw_label": the exact line-item label text, e.g. "Total revenues",
  "year": the four-digit fiscal year the value covers, as an integer,
  "quarter": 1, 2, 3, or 4 if the value is for a specific quarter, or null for a full fiscal year or year-to-date figure,
  "is_ttm": true only if the value is explicitly a trailing-twelve-month figure, else false,
  "value": the numeric value as a JSON number (strip "$" and ",", parentheses mean negative),
  "units": one of "USD_millions", "USD_thousands", "USD", "percent", "shares_millions", "shares_thousands", "shares", "per_share",
  "is_gaap": false only if the row label itself says "non-GAAP" or "adjusted", or the table is a non-GAAP reconciliation and this is the adjusted figure, else true,
  "is_restated": true only if the column header or a footnote marker on this row says "restated" or "as restated", else false
}

Rules:
1. Use the table's column headers to determine each column's period. Do not guess a period that the headers do not indicate.
2. If a row's period cannot be determined with confidence, omit that fact rather than guessing.
3. If a cell value is not a number (dashes, blank, "N/A", "*"), omit that fact rather than inventing a value.
4. Return JSON only, no prose outside the JSON object.
'''


# =============================================================================
# Public entry points
# =============================================================================

def extract_facts(
    table: ParsedTable,
    classification: TableClassification,
) -> List[ExtractedFact]:
    """Extract numeric facts from one classified table.

    Args:
        table: A table extracted by ingest/pdf_parser.py.
        classification: The category assigned by ingest/table_classifier.py
            for this same table.

    Returns:
        Validated facts with full provenance. Empty if the table is
        TableCategory.OTHER, if the LLM call or JSON parse fails, or if no
        row passes validation. A partial or failed extraction never raises;
        it produces fewer facts, which the online Verifier later surfaces
        as insufficient_data rather than a wrong answer.
    """
    if classification.category == TableCategory.OTHER:
        return []

    user_msg = (
        f"Table ID: {table.table_id}\n"
        f"Page: {table.page_number}\n"
        f"Classified category: {classification.category.value}\n"
        f"Caption: {table.caption or '(none)'}\n\n"
        f"Table contents:\n{render_table_as_text(table)}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=EXTRACTION_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        if response.stop_reason == "max_tokens":
            log.warning(
                "Table %s: extraction response hit the %d-token cap and was "
                "truncated (table has %d rows); raise EXTRACTION_MAX_TOKENS "
                "rather than trusting a partial JSON parse.",
                table.table_id, EXTRACTION_MAX_TOKENS, table.n_rows,
            )
            return []
        raw = response.content[0].text
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        raw_rows = data["facts"]
    except Exception as exc:
        log.warning("Table %s: fact extraction failed: %s", table.table_id, exc)
        return []

    facts: List[ExtractedFact] = []
    units_seen: dict[str, int] = {}
    for row in raw_rows:
        candidate = _validate_row(table.table_id, row)
        if candidate is None:
            continue
        units_seen[candidate.units] = units_seen.get(candidate.units, 0) + 1
        facts.append(ExtractedFact(
            metric_raw_label=candidate.metric_raw_label,
            year=candidate.year,
            quarter=candidate.quarter,
            is_ttm=candidate.is_ttm,
            value=candidate.value,
            units=candidate.units,
            is_gaap=candidate.is_gaap,
            is_restated=candidate.is_restated,
            filing_id=table.filing_id,
            page_number=table.page_number,
            table_id=table.table_id,
            row_id=candidate.row_id,
        ))

    if len(units_seen) > 3:
        log.info(
            "Table %s: %d distinct unit types across extracted facts (%s); "
            "expected for tables mixing dollar, share-count, and percent rows.",
            table.table_id, len(units_seen), units_seen,
        )

    log.info(
        "Table %s (%s): extracted %d facts from %d candidate rows",
        table.table_id, classification.category.value, len(facts), len(raw_rows),
    )
    return facts


def extract_facts_from_tables(
    tables: List[ParsedTable],
    classifications: List[TableClassification],
) -> List[ExtractedFact]:
    """Extract facts from a batch of already-classified tables.

    Args:
        tables: Tables extracted by ingest/pdf_parser.py.
        classifications: One classification per table, same order, as
            produced by ingest/table_classifier.classify_tables().

    Returns:
        All extracted facts across every table, concatenated.
    """
    all_facts: List[ExtractedFact] = []
    for table, classification in zip(tables, classifications):
        all_facts.extend(extract_facts(table, classification))
    return all_facts


# =============================================================================
# Pass two: rule-based validation
# =============================================================================

def _validate_row(table_id: str, row: dict) -> Optional[_RawFactRow]:
    """Validate one LLM-extracted row: shape, numeric value, plausible period.

    Args:
        table_id: Source table, used only for log context.
        row: One item from the LLM's "facts" array.

    Returns:
        The validated row, or None if it fails shape validation or the
        rule-based sanity checks (value is a real number, year is
        plausible, quarter is 1-4 or null).
    """
    try:
        candidate = _RawFactRow(**row)
    except (ValidationError, TypeError) as exc:
        log.warning("Table %s: dropping malformed row %r: %s", table_id, row, exc)
        return None

    if not (MIN_PLAUSIBLE_YEAR <= candidate.year <= MAX_PLAUSIBLE_YEAR):
        log.warning(
            "Table %s: dropping row_id=%d, implausible year %d",
            table_id, candidate.row_id, candidate.year,
        )
        return None

    return candidate


# =============================================================================
# CLI entry point (manual testing / one-off extraction)
# =============================================================================

if __name__ == "__main__":
    import argparse

    from app.ingest.pdf_parser import parse_filing
    from app.ingest.table_classifier import classify_tables

    ap = argparse.ArgumentParser(
        description="Parse a filing PDF, classify its tables, and extract facts.",
    )
    ap.add_argument("pdf_path", help="Path to filing PDF")
    ap.add_argument(
        "--filing-id",
        required=True,
        help="Stable identifier, e.g. TSLA-10K-2025-12-31",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parsed = parse_filing(args.pdf_path, args.filing_id)
    classifications = classify_tables(parsed.tables)
    facts = extract_facts_from_tables(parsed.tables, classifications)
    print(json.dumps([f.model_dump() for f in facts], indent=2))
