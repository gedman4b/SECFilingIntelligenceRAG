"""
ingest/table_classifier.py

Table classification: LLM with schema-constrained output, run once per
parsed table at ingestion time.

Design principle
-----------------
This is the first place an LLM touches filing content, and it runs
offline, once per table, not online per query. The interpretation cost
(what statement is this, what does the caption mean) is expensive and
belongs here rather than in the query-time path (Part 1 of the write-up:
"parse and structure the corpus once, up front, so that query time is a
lookup against a clean fact store"). Classification failures fail closed
to 'other' rather than guessing a wrong category, since a misclassified
income statement flowing into fact_extractor.py would mislabel every fact
drawn from it.

Category taxonomy matches the write-up's offline pipeline stage [2]:
income_stmt | balance_sheet | cash_flow | non_gaap_reconciliation |
segment | other.

Dependencies
------------
anthropic  (LLM call)
pydantic >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import List, Optional

import anthropic
from pydantic import BaseModel

from app.ingest.pdf_parser import ParsedTable
from app.instrumentation import log_latency

log = logging.getLogger(__name__)

client = anthropic.Anthropic()


# =============================================================================
# Output schema
# =============================================================================

class TableCategory(str, Enum):
    INCOME_STMT = "income_stmt"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    NON_GAAP_RECONCILIATION = "non_gaap_reconciliation"
    SEGMENT = "segment"
    OTHER = "other"


class TableClassification(BaseModel):
    """Classification result for a single parsed table."""
    table_id: str
    category: TableCategory
    rationale: Optional[str] = None


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = '''You classify a single table extracted from an SEC filing (10-K or 10-Q).

Return a JSON object with exactly these fields:
{
  "category": one of "income_stmt", "balance_sheet", "cash_flow", "non_gaap_reconciliation", "segment", "other",
  "rationale": a one-sentence justification for the category
}

Category definitions:
- income_stmt: revenue, costs and expenses, operating income, net income.
- balance_sheet: assets, liabilities, and stockholders' equity at a point in time.
- cash_flow: cash flows from operating, investing, and financing activities.
- non_gaap_reconciliation: reconciles a GAAP measure to a non-GAAP (adjusted) measure.
- segment: revenue or operating results broken out by business segment or geography.
- other: anything that is not one of the above (share count tables, tax rate tables, unrelated layout tables).

Rules:
1. The table's caption, if present, is the strongest signal. Captions like
   "CONSOLIDATED STATEMENTS OF OPERATIONS" or "CONSOLIDATED BALANCE SHEETS"
   should be trusted over row-level inference.
2. If the table's category is genuinely ambiguous, choose "other" rather than guessing.
3. Return JSON only, no prose outside the JSON object.
'''


# =============================================================================
# Public entry points
# =============================================================================

@log_latency(log)
def classify_table(table: ParsedTable) -> TableClassification:
    """Classify a single parsed table into the offline-pipeline taxonomy.

    Args:
        table: A table extracted by ingest/pdf_parser.py.

    Returns:
        The classification. On any LLM or parse failure, falls back to
        TableCategory.OTHER with the failure recorded in rationale rather
        than guessing a specific statement type.
    """
    user_msg = (
        f"Table ID: {table.table_id}\n"
        f"Page: {table.page_number}\n"
        f"Caption: {table.caption or '(none)'}\n\n"
        f"Table contents:\n{render_table_as_text(table)}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return TableClassification(
            table_id=table.table_id,
            category=TableCategory(data["category"]),
            rationale=data.get("rationale"),
        )
    except Exception as exc:  # fail closed: unclassified, not a guess
        log.warning("Table %s: classification failed: %s", table.table_id, exc)
        return TableClassification(
            table_id=table.table_id,
            category=TableCategory.OTHER,
            rationale=f"classification_failed: {exc}",
        )


def classify_tables(tables: List[ParsedTable]) -> List[TableClassification]:
    """Classify a batch of parsed tables.

    A failure classifying one table does not abort the batch; it is
    recorded as TableCategory.OTHER for that table only, following the
    same never-abort-on-one-failure pattern as pdf_parser.py.

    Args:
        tables: Tables extracted by ingest/pdf_parser.py.

    Returns:
        One TableClassification per input table, same order.
    """
    return [classify_table(table) for table in tables]


# =============================================================================
# Helpers
# =============================================================================

def render_table_as_text(table: ParsedTable) -> str:
    """Reconstruct a parsed table's flat cell list into a readable grid.

    Args:
        table: A table extracted by ingest/pdf_parser.py.

    Returns:
        The table rendered as pipe-delimited rows, in row/column order.
    """
    grid = {(cell.row, cell.col): cell.value for cell in table.cells}
    lines = []
    for r in range(table.n_rows):
        row_cells = [grid.get((r, c), "") for c in range(table.n_cols)]
        lines.append(" | ".join(row_cells))
    return "\n".join(lines)


# =============================================================================
# CLI entry point (manual testing / one-off classification)
# =============================================================================

if __name__ == "__main__":
    import argparse

    from app.ingest.pdf_parser import parse_filing

    ap = argparse.ArgumentParser(
        description="Parse a filing PDF and classify each extracted table.",
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
    print(json.dumps([c.model_dump() for c in classifications], indent=2))
