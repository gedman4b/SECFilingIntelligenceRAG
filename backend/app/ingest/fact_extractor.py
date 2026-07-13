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
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from app.ingest.pdf_parser import ParsedTable
from app.ingest.table_classifier import TableCategory, TableClassification, render_table_as_text
from app.instrumentation import log_latency

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
# Filing-level context
# =============================================================================

class FilingContext(BaseModel):
    """Filing-level metadata needed to disambiguate period columns that a
    table's own headers don't state explicitly.

    A 10-Q's condensed statements routinely show bare-year column headers
    (e.g. "2026 2025") with no "Three Months Ended" qualifier anywhere in
    the table itself; that qualifier sits in surrounding page text that
    pdf_parser.py's caption heuristic does not reliably capture. Rather
    than have the LLM guess from ambiguous in-table text, the orchestrator
    already knows the filing's form type and fiscal period with certainty
    (read from the filing's cover page, see ingest/run_ingestion.py) and
    passes it in directly.
    """
    form_type: str  # '10-K' | '10-Q' | '10-K/A' | '10-Q/A'
    fiscal_year: int
    fiscal_quarter: Optional[int] = None  # None for a 10-K


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
    is_ytd: bool = False
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
    is_ytd: bool = False
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

    @model_validator(mode="after")
    def _ytd_requires_quarter(self) -> "_RawFactRow":
        if self.is_ytd and self.quarter is None:
            raise ValueError("is_ytd=true requires a non-null quarter (the ending quarter of the YTD period)")
        return self


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
  "quarter": 1, 2, 3, or 4 if the value covers that fiscal quarter (whether alone or as part of a year-to-date range ending in it), or null only for a full fiscal year figure,
  "is_ttm": true only if the value is explicitly a trailing-twelve-month figure, else false,
  "is_ytd": true if the value is a year-to-date cumulative figure (e.g. "six months ended", "nine months ended") rather than a single discrete quarter or a full fiscal year, else false. When true, "quarter" is the ending quarter of the cumulative period, e.g. "six months ended June 30" is quarter=2, is_ytd=true,
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

@log_latency(log)
def extract_facts(
    table: ParsedTable,
    classification: TableClassification,
    filing_context: FilingContext,
) -> List[ExtractedFact]:
    """Extract numeric facts from one classified table.

    Args:
        table: A table extracted by ingest/pdf_parser.py.
        classification: The category assigned by ingest/table_classifier.py
            for this same table.
        filing_context: Form type and fiscal period of the source filing,
            used to disambiguate bare-year column headers instead of
            leaving the LLM to guess from in-table text alone.

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
        f"{_period_guidance(filing_context)}\n\n"
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
            is_ytd=candidate.is_ytd,
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
    filing_context: FilingContext,
) -> List[ExtractedFact]:
    """Extract facts from a batch of already-classified tables.

    Args:
        tables: Tables extracted by ingest/pdf_parser.py.
        classifications: One classification per table, same order, as
            produced by ingest/table_classifier.classify_tables().
        filing_context: Form type and fiscal period of the source filing,
            shared across every table in one filing.

    Returns:
        All extracted facts across every table, concatenated.
    """
    all_facts: List[ExtractedFact] = []
    for table, classification in zip(tables, classifications):
        all_facts.extend(extract_facts(table, classification, filing_context))
    return all_facts


def _period_guidance(filing_context: FilingContext) -> str:
    """Build the period-disambiguation paragraph injected into every
    extraction call, so the LLM never has to guess a filing's period type
    from ambiguous in-table text when the orchestrator already knows it.

    Args:
        filing_context: Form type and fiscal period of the source filing.

    Returns:
        A short paragraph to prepend to the user message.
    """
    period_desc = (
        f"fiscal quarter {filing_context.fiscal_quarter} of fiscal year "
        f"{filing_context.fiscal_year}"
        if filing_context.fiscal_quarter is not None
        else f"the full fiscal year {filing_context.fiscal_year}"
    )
    guidance = (
        f"This table is from a {filing_context.form_type} covering "
        f"{period_desc}."
    )
    if filing_context.fiscal_quarter is not None:
        guidance += (
            " If a column header shows only a bare year with no period "
            "qualifier (no \"twelve months\", \"fiscal year\", \"annual\"), "
            f"assume that column represents fiscal quarter "
            f"{filing_context.fiscal_quarter} for that year, not the full "
            "year. 10-Q tables routinely show quarterly comparatives under "
            "bare-year headers, with the \"three/six/nine months ended\" "
            "qualifier appearing only in surrounding page text, not in the "
            "table itself. A single table can also show BOTH a discrete "
            "quarter column and a separate year-to-date column covering "
            "multiple quarters (e.g. \"Three Months Ended\" next to \"Six "
            "Months Ended\") -- these are two different facts, not "
            "duplicates: set is_ytd=false and quarter equal to the fiscal "
            "quarter above for the discrete column, and is_ytd=true with "
            "quarter equal to the ENDING quarter of the range for the "
            "cumulative column. Never label a year-to-date figure with "
            "quarter=null; that value is reserved exclusively for a true "
            "full fiscal year figure."
        )
    else:
        guidance += (
            " A bare-year column header in a 10-K normally represents the "
            "full fiscal year unless the table itself states otherwise."
        )
    return guidance


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
    ap.add_argument("--form-type", required=True, help="e.g. 10-K, 10-Q, 10-K/A")
    ap.add_argument("--fiscal-year", required=True, type=int)
    ap.add_argument("--fiscal-quarter", type=int, default=None, help="Omit for a 10-K")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parsed = parse_filing(args.pdf_path, args.filing_id)
    classifications = classify_tables(parsed.tables)
    filing_context = FilingContext(
        form_type=args.form_type, fiscal_year=args.fiscal_year,
        fiscal_quarter=args.fiscal_quarter,
    )
    facts = extract_facts_from_tables(parsed.tables, classifications, filing_context)
    print(json.dumps([f.model_dump() for f in facts], indent=2))
