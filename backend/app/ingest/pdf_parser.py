"""
ingest/pdf_parser.py

Layout-aware PDF parser for SEC filings.

Extracts tables (row/column structure preserved) and narrative prose sections
(chunked by SEC-standard section) from a filing PDF.

Design principle
----------------
This stage does NOT interpret. It only extracts structural elements with full
provenance. Deciding what statement a table represents, what a label means,
what canonical metric it maps to, or how to reconcile GAAP vs non-GAAP is the
job of downstream stages (table_classifier, fact_extractor, canonicalizer).

Keeping the parser strictly structural has three benefits:
  1. It is deterministic and cheap to re-run on the same corpus.
  2. It can be tested in isolation against ground-truth extractions.
  3. Interpretation failures downstream do not require re-parsing the PDF.

Dependencies
------------
pdfplumber >= 0.11
pydantic   >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# =============================================================================
# Output schemas
# =============================================================================

class TableCell(BaseModel):
    """A single cell in a parsed table. Position preserved. Value is raw text."""
    row: int
    col: int
    value: str
    is_header: bool = False


class ParsedTable(BaseModel):
    """A single table extracted from a PDF page.

    Structure preserved. No interpretation yet.
    """
    filing_id: str
    table_id: str            # stable: f"{filing_id}::page{N}::table{M}"
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, top, x1, bottom)
    n_rows: int
    n_cols: int
    cells: List[TableCell]
    caption: Optional[str] = None            # text just above the table


class ProseSection(BaseModel):
    """A narrative prose section (MD&A, Risk Factors, Business, etc.)."""
    filing_id: str
    section_id: str
    section_type: str        # "mdna" | "risk_factors" | "business" | ...
    heading: str
    page_start: int
    page_end: int
    text: str


class ParsedFiling(BaseModel):
    """Complete parser output for a single filing."""
    filing_id: str
    filing_path: str
    n_pages: int
    tables: List[ParsedTable]
    prose_sections: List[ProseSection]
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# SEC section-heading patterns
# =============================================================================
# Matches the standard 10-K and 10-Q "Item N. Title" headings. 10-K uses
# Item 7 for MD&A; 10-Q uses Item 2. The [27] alternation handles both.

SECTION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "risk_factors",
        re.compile(r"^\s*Item\s+1A[.\s]+Risk\s+Factors", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "mdna",
        re.compile(
            r"^\s*Item\s+[27][.\s]+Management['s\s]*Discussion\s+and\s+Analysis",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "business",
        re.compile(r"^\s*Item\s+1[.\s]+Business", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "quantitative_qualitative",
        re.compile(
            r"^\s*Item\s+7A[.\s]+Quantitative\s+and\s+Qualitative",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "financial_statements",
        re.compile(
            r"^\s*Item\s+8[.\s]+Financial\s+Statements",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "controls",
        re.compile(
            r"^\s*Item\s+9A[.\s]+Controls\s+and\s+Procedures",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]


# Financial numeric cell recognizer. Handles:
#   1,234           1,234.56       (1,234)       (1,234.56)
#   $1,234          $1,234.56      -1,234
#   12.5%           1234           1              (0.02)
_NUMERIC_RE = re.compile(
    r"^\s*\(?\s*\$?\s*-?\d[\d,]*(?:\.\d+)?\s*\)?\s*%?\s*$"
)


# =============================================================================
# Public entry point
# =============================================================================

def parse_filing(pdf_path: str, filing_id: str) -> ParsedFiling:
    """Parse a single SEC filing PDF into structural elements.

    Args:
        pdf_path: absolute path to the filing PDF.
        filing_id: stable identifier, e.g. "TSLA-10K-2024-12-31".

    Returns:
        ParsedFiling with tables and prose sections extracted.

    Raises:
        FileNotFoundError: if pdf_path does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Filing not found: {pdf_path}")

    warnings: List[str] = []
    tables: List[ParsedTable] = []
    prose_sections: List[ProseSection] = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        log.info("Opened %s: %d pages", filing_id, n_pages)

        # Early detection: image-only filings need OCR before this parser
        # can extract anything useful. Report cleanly rather than returning
        # a silently empty result.
        text_sample = [
            (pdf.pages[i].extract_text() or "").strip()
            for i in range(min(5, n_pages))
        ]
        if all(len(t) < 20 for t in text_sample):
            warnings.append(
                "First 5 pages contain no extractable text; filing may be "
                "image-only or encrypted. OCR preprocessing required before "
                "this parser can extract structured content."
            )
            return ParsedFiling(
                filing_id=filing_id,
                filing_path=str(path),
                n_pages=n_pages,
                tables=[],
                prose_sections=[],
                warnings=warnings,
            )

        # ---------- Pass 1: tables per page ----------
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                page_tables = _extract_tables_from_page(page, page_idx, filing_id)
                tables.extend(page_tables)
            except Exception as exc:  # never abort on a single-page failure
                msg = f"page {page_idx}: table extraction failed: {exc}"
                warnings.append(msg)
                log.warning("%s %s", filing_id, msg)

        # ---------- Pass 2: prose sections (whole document) ----------
        try:
            prose_sections = _extract_prose_sections(pdf, filing_id)
        except Exception as exc:
            msg = f"prose section extraction failed: {exc}"
            warnings.append(msg)
            log.warning("%s %s", filing_id, msg)

    log.info(
        "%s: extracted %d tables and %d prose sections",
        filing_id, len(tables), len(prose_sections),
    )

    return ParsedFiling(
        filing_id=filing_id,
        filing_path=str(path),
        n_pages=n_pages,
        tables=tables,
        prose_sections=prose_sections,
        warnings=warnings,
    )


# =============================================================================
# Table extraction helpers
# =============================================================================

def _extract_tables_from_page(
    page, page_number: int, filing_id: str,
) -> List[ParsedTable]:
    """Extract tables from a single page with row/column structure preserved."""
    results: List[ParsedTable] = []

    # find_tables() returns Table objects with bbox metadata; extract_tables()
    # only returns the row data. We want bbox for provenance and for looking
    # up the table caption.
    found = page.find_tables()

    for tbl_idx, tbl in enumerate(found, start=1):
        rows = tbl.extract()
        if not rows or not _looks_like_financial_table(rows):
            continue

        header_row = _detect_header_row(rows)

        cells: List[TableCell] = []
        n_cols = max(len(r) for r in rows) if rows else 0
        for r_idx, row in enumerate(rows):
            for c_idx, raw_value in enumerate(row):
                cleaned = _clean_cell(raw_value)
                if cleaned is None:
                    continue
                cells.append(TableCell(
                    row=r_idx,
                    col=c_idx,
                    value=cleaned,
                    is_header=(r_idx == header_row),
                ))

        caption = _find_table_caption(page, tbl.bbox)
        table_id = f"{filing_id}::page{page_number}::table{tbl_idx}"

        results.append(ParsedTable(
            filing_id=filing_id,
            table_id=table_id,
            page_number=page_number,
            bbox=tbl.bbox,
            n_rows=len(rows),
            n_cols=n_cols,
            cells=cells,
            caption=caption,
        ))

    return results


def _looks_like_financial_table(rows: List[List[Optional[str]]]) -> bool:
    """Filter out layout tables and TOCs.

    Real financial tables have at least 2 rows, at least 2 columns, and a
    meaningful fraction of numeric-looking cells. A 20 percent numeric-cell
    threshold catches financial statements while rejecting most TOCs, page
    numbers, and layout tables used for narrative formatting.
    """
    if len(rows) < 2:
        return False
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols < 2:
        return False

    total = 0
    numeric = 0
    for row in rows:
        for cell in row:
            if cell is None or not str(cell).strip():
                continue
            total += 1
            if _cell_is_numeric(str(cell)):
                numeric += 1

    if total == 0:
        return False
    return (numeric / total) >= 0.20


def _cell_is_numeric(text: str) -> bool:
    """Recognize a financial cell value: 1,234.56, (1,234), $1,234, 12.5%, etc."""
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_NUMERIC_RE.match(stripped))


def _detect_header_row(rows: List[List[Optional[str]]]) -> int:
    """Return the row index that looks like the header row.

    Headers are characterized by few numeric cells and typically contain
    period labels like "Three Months Ended", "2024", "Q1". Header row is
    almost always in the first three rows of a financial table.
    """
    if not rows:
        return -1
    for r_idx, row in enumerate(rows[:3]):
        non_empty = sum(1 for c in row if c and str(c).strip())
        if non_empty == 0:
            continue
        numeric = sum(1 for c in row if c and _cell_is_numeric(str(c)))
        if numeric / non_empty < 0.25:
            return r_idx
    return 0


def _clean_cell(value) -> Optional[str]:
    """Normalize whitespace. Return None for empty cells."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def _find_table_caption(page, bbox) -> Optional[str]:
    """Look above the table bbox for a heading line that names it.

    Financial tables typically carry headings like
      "CONSOLIDATED STATEMENTS OF OPERATIONS"
      "Reconciliation of GAAP to Non-GAAP Financial Measures"
    which downstream stages use as a strong signal for table classification.
    """
    try:
        x0, top, x1, _bottom = bbox
        strip = page.within_bbox((x0, max(0, top - 60), x1, top))
        text = strip.extract_text() or ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            return None
        candidate = lines[-1]
        # Reject bare page numbers and very short artifacts
        if len(candidate) > 3 and not candidate.isdigit():
            return candidate[:200]
    except Exception:
        # Caption is a helpful signal but not required. Failing here is fine.
        pass
    return None


# =============================================================================
# Prose section extraction
# =============================================================================

def _extract_prose_sections(pdf, filing_id: str) -> List[ProseSection]:
    """Locate SEC-standard prose sections and extract their text.

    Strategy: build a page-by-page text index, find every occurrence of each
    known section heading pattern, then keep only the LAST occurrence of each
    section type. This heuristic skips the table-of-contents mention of each
    section and finds the actual body.
    """
    per_page_text: List[Tuple[int, str]] = []
    for page_idx, page in enumerate(pdf.pages, start=1):
        per_page_text.append((page_idx, page.extract_text() or ""))

    # Find every candidate section start.
    all_starts: List[Tuple[int, str, str]] = []  # (page, section_type, heading)
    for page_idx, text in per_page_text:
        for section_type, pattern in SECTION_PATTERNS:
            for match in pattern.finditer(text):
                start = match.start()
                line_end = text.find("\n", start)
                heading = (
                    text[start:line_end].strip() if line_end != -1
                    else text[start:].strip()
                )
                all_starts.append((page_idx, section_type, heading))

    # Keep only the LAST occurrence of each section type (skips TOC hits).
    last_of_type = {}
    for page_idx, section_type, heading in all_starts:
        last_of_type[section_type] = (page_idx, section_type, heading)

    final_starts = sorted(last_of_type.values(), key=lambda x: x[0])

    sections: List[ProseSection] = []
    for i, (start_page, section_type, heading) in enumerate(final_starts):
        end_page = (
            final_starts[i + 1][0] - 1 if i + 1 < len(final_starts)
            else len(per_page_text)
        )
        buf = [t for p, t in per_page_text if start_page <= p <= end_page]
        section_text = "\n".join(buf).strip()
        if not section_text:
            continue

        sections.append(ProseSection(
            filing_id=filing_id,
            section_id=f"{filing_id}::section::{section_type}",
            section_type=section_type,
            heading=heading,
            page_start=start_page,
            page_end=end_page,
            text=section_text,
        ))

    return sections


# =============================================================================
# CLI entry point (manual testing / one-off ingestion)
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Parse an SEC filing PDF into structured tables and prose.",
    )
    ap.add_argument("pdfs", help="Path to filing PDF")
    ap.add_argument(
        "--filing-id",
        required=True,
        help="Stable identifier, e.g. TSLA-10K-2024-12-31",
    )
    ap.add_argument(
        "--out",
        default="-",
        help="Output JSON path, or '-' for stdout (default)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parsed = parse_filing(args.pdf, args.filing_id)
    payload = parsed.model_dump()

    if args.out == "-":
        print(json.dumps(payload, indent=2))
    else:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(
            f"Wrote {args.out}: {len(parsed.tables)} tables, "
            f"{len(parsed.prose_sections)} sections, "
            f"{len(parsed.warnings)} warnings"
        )