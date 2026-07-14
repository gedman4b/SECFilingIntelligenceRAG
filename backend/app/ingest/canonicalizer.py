"""
ingest/canonicalizer.py

Metric canonicalization: deterministic mapping from raw filing labels to a
fixed canonical metric registry.

Design principle
-----------------
This module is intentionally not an LLM stage, unlike table_classifier.py
and fact_extractor.py. agents/fact_retriever.py calls
resolve_canonical_metric() directly from the online query path, and the
Fact Retriever is a declared LLM-free, deterministic stage (Part 4 of the
write-up: "It is a database query... Wrapping a database query in an LLM
adds cost, latency, and hallucination risk"). A canonicalizer that used an
LLM, or any fuzzy/substring matching, would smuggle exactly the kind of
near-miss ambiguity embeddings are banned from the numeric path for back
in through a side door: "Net income" and "Net income attributable to
common stockholders" must never collapse into one canonical ID (Part 1 of
the write-up, Part 3 Guard 1).

So resolution here is exact-match-after-normalization against a curated
registry (Part 3 of the write-up: "Raw labels are mapped to a canonical
metric registry"). A label that is not an exact match, after normalizing
case, whitespace, curly quotes, and footnote markers, is left unresolved
and flagged rather than guessed (Part 3 Guard 2: "the metric canonicalizer
... emits an ambiguity_flag rather than force-mapping"). Facts with an
unresolved metric are still stored, per the write-up, so the Verifier can
surface the uncertainty instead of the fact silently disappearing.

The registry below is grounded in labels actually observed in the real
Tesla and Apple filings in ingest/pdfs/, not guessed: "Total revenues"
(Tesla) and "Total net sales" (Apple) are genuinely different company
phrasings for the same canonical metric, which is the whole reason this
stage exists.

Dependencies
------------
pydantic >= 2.0

Author: Scott Josephson  |  Deloitte SEC Filing Intelligence take-home
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.ingest.fact_extractor import ExtractedFact
from app.store.db import FactRecord

log = logging.getLogger(__name__)

# Sentinel canonical id for a fact whose label did not match the registry.
# Never a real metric_id, so a query for an actual metric can never
# accidentally retrieve it; the fact stays stored and visible via its
# ambiguity_flags rather than being dropped.
UNRESOLVED_METRIC_ID = "UNRESOLVED"


# =============================================================================
# Canonical metric registry
# =============================================================================

class CanonicalMetric(BaseModel):
    """One entry in the canonical metric registry."""
    metric_id: str
    display_name: str
    statement: str  # 'income_stmt' | 'balance_sheet' | 'cash_flow'
    known_labels: List[str]  # raw label variants; normalized at load time
    # True for cost/expense line items, where a decrease is an improvement
    # and an increase is deterioration -- the opposite of a revenue or
    # profit metric. Curated by hand, not inferred from label text, so the
    # ranking question_type (agents/numerical_reasoner.py) can score
    # "deteriorated most" deterministically instead of guessing from the
    # metric's name.
    is_expense: bool = False


CANONICAL_METRICS: List[CanonicalMetric] = [
    CanonicalMetric(
        metric_id="METRIC_TOTAL_REVENUE",
        display_name="Total revenue",
        statement="income_stmt",
        known_labels=[
            "Total revenues", "Total net sales", "Net revenues",
            "total revenue", "revenue", "revenues", "net sales", "sales",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_COST_OF_REVENUE",
        display_name="Total cost of revenue",
        statement="income_stmt",
        known_labels=[
            "Total cost of revenues", "Total cost of sales",
            "cost of revenue", "cost of goods sold", "cogs",
        ],
        is_expense=True,
    ),
    CanonicalMetric(
        metric_id="METRIC_GROSS_PROFIT",
        display_name="Gross profit",
        statement="income_stmt",
        known_labels=["Gross profit", "Gross margin"],
    ),
    CanonicalMetric(
        metric_id="METRIC_OPERATING_EXPENSES",
        display_name="Total operating expenses",
        statement="income_stmt",
        known_labels=["Total operating expenses", "operating expenses"],
        is_expense=True,
    ),
    CanonicalMetric(
        metric_id="METRIC_SGA",
        display_name="Selling, general and administrative expenses",
        statement="income_stmt",
        known_labels=[
            "Selling, general and administrative", "sg&a", "sga",
            "selling, general and administrative expenses",
        ],
        is_expense=True,
    ),
    CanonicalMetric(
        metric_id="METRIC_OPERATING_INCOME",
        display_name="Operating income",
        statement="income_stmt",
        known_labels=["Income from operations", "Operating income", "operating profit"],
    ),
    CanonicalMetric(
        metric_id="METRIC_NET_INCOME",
        display_name="Net income",
        statement="income_stmt",
        # Bare "profit" added after a real live-tested gap: "Tesla's
        # change in profit" did not resolve, even though bare "profit" in
        # everyday financial usage means the bottom line (net income),
        # not gross or operating profit -- those are always said with
        # their qualifying word ("gross profit", "operating profit", the
        # latter already a known_label for METRIC_OPERATING_INCOME
        # above). Safe: no raw label extracted from any filing in this
        # corpus is bare "profit" (confirmed by direct query), only
        # qualified variants like "Gross profit total automotive", so
        # this can never collide with an actual filing's own label text
        # the way "Automotive sales" did.
        known_labels=["Net income", "Net income (loss)", "Net earnings", "earnings", "profit"],
    ),
    CanonicalMetric(
        metric_id="METRIC_NET_INCOME_ATTRIBUTABLE_COMMON",
        display_name="Net income attributable to common stockholders",
        statement="income_stmt",
        # Deliberately its own metric_id, not a synonym of METRIC_NET_INCOME:
        # it is a distinct dollar figure (Part 3 of the write-up's own
        # canonicalization example).
        known_labels=[
            "Net income attributable to common stockholders",
            "net income attributable to shareholders",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_TOTAL_ASSETS",
        display_name="Total assets",
        statement="balance_sheet",
        known_labels=["Total assets", "assets"],
    ),
    CanonicalMetric(
        metric_id="METRIC_TOTAL_LIABILITIES",
        display_name="Total liabilities",
        statement="balance_sheet",
        known_labels=["Total liabilities", "liabilities"],
    ),
    CanonicalMetric(
        metric_id="METRIC_TOTAL_EQUITY",
        display_name="Total stockholders' equity",
        statement="balance_sheet",
        known_labels=[
            "Total stockholders' equity",
            "Total shareholders' equity",
            "stockholders' equity", "shareholders' equity", "equity",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_CASH_AND_EQUIVALENTS",
        display_name="Cash and cash equivalents",
        statement="balance_sheet",
        known_labels=["Cash and cash equivalents", "cash and equivalents", "cash"],
    ),
    CanonicalMetric(
        metric_id="METRIC_OPERATING_CASH_FLOW",
        display_name="Net cash provided by operating activities",
        statement="cash_flow",
        known_labels=[
            "Net cash provided by operating activities",
            "Cash generated by operating activities",
            "operating cash flow", "cash flow from operations", "cash from operations",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_EPS_BASIC",
        display_name="Basic earnings per share",
        statement="income_stmt",
        # Deliberately does NOT include the bare label "Basic": real 10-K
        # tables use that exact text for both this metric (a per-share
        # dollar figure) and weighted-average basic share count (a share
        # count), under different section headers. See
        # UNITS_DISAMBIGUATED_LABELS below for how the bare form is
        # resolved using units, not label text alone.
        known_labels=[
            "basic earnings per share", "basic eps", "basic net income per share",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_EPS_DILUTED",
        display_name="Diluted earnings per share",
        statement="income_stmt",
        known_labels=[
            "diluted earnings per share", "diluted eps", "diluted net income per share",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_WEIGHTED_AVG_SHARES_BASIC",
        display_name="Weighted average basic shares outstanding",
        statement="income_stmt",
        known_labels=[
            "weighted average basic shares outstanding", "weighted-average basic shares outstanding",
            "basic weighted average shares", "weighted average shares outstanding, basic",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_WEIGHTED_AVG_SHARES_DILUTED",
        display_name="Weighted average diluted shares outstanding",
        statement="income_stmt",
        known_labels=[
            "weighted average diluted shares outstanding", "weighted-average diluted shares outstanding",
            "diluted weighted average shares", "weighted average shares outstanding, diluted",
        ],
    ),
    # Tesla's segment revenue breakdown (10-K/10-Q "Revenues" footnote
    # table). Added after a real live-tested gap: "revenue from car sales"
    # did not match the exact filing text "Automotive sales", and the
    # numeric path's raw-label fallback (fact_retriever.py) is intentionally
    # exact-match only, never fuzzy, so it correctly found nothing rather
    # than guess.
    #
    # "Automotive sales" and "Automotive leasing" are deliberately NOT
    # canonical metrics here, despite being the two most obviously-useful
    # ones. Tesla's own income statement (confirmed against the actual
    # PDF, TSLA-10Q-2026-03-31 page 5) reuses the EXACT text "Automotive
    # sales" for two different line items -- once under "Revenues" ($15,473
    # for Q1 2026) and again, verbatim, under "Cost of revenues" ($12,616
    # for the same quarter) -- and same for "Automotive leasing" and
    # "Services and other". A canonical metric_id has no way to know which
    # of the two a given row came from; the extraction pipeline does not
    # currently capture which section of the table a row belongs to. Live
    # testing caught this the hard way: an earlier version of this registry
    # force-mapped bare "Automotive sales" to one metric_id, which silently
    # merged the revenue and cost-of-revenue figures and surfaced as a
    # "conflicting values" verifier error instead of a clean answer -- correct
    # fail-closed behavior, but for the wrong reason (it made a real,
    # resolvable-in-principle ambiguity look identical to the case where
    # nothing is known at all). Properly resolving this needs a
    # section-aware signal captured at extraction time (analogous to how
    # is_expense and the Basic/Diluted units-disambiguation already solve
    # different versions of this same "same label, two meanings" problem),
    # which is a real, scoped follow-up, not a quick synonym fix. Until
    # then, both stay unresolved and fail closed rather than guess.
    #
    # "Automotive regulatory credits" and "Total automotive revenues" are
    # safe: confirmed via direct query that each has exactly one value per
    # (company, year, quarter) across every filing and table -- no
    # cost-of-revenue line reuses either exact text.
    CanonicalMetric(
        metric_id="METRIC_AUTOMOTIVE_REGULATORY_CREDITS",
        display_name="Automotive regulatory credits",
        statement="income_stmt",
        known_labels=[
            "Automotive regulatory credits", "regulatory credits", "regulatory credit revenue",
            "automotive regulatory credit revenue",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_TOTAL_AUTOMOTIVE_REVENUE",
        display_name="Total automotive revenues",
        statement="income_stmt",
        known_labels=[
            "Total automotive revenues", "automotive revenue", "automotive segment revenue",
            "total automotive revenue",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_ENERGY_SALES",
        display_name="Energy generation and storage sales",
        statement="income_stmt",
        known_labels=[
            "Energy generation and storage sales", "energy storage sales", "energy hardware sales",
            "energy sales revenue",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_ENERGY_LEASING",
        display_name="Energy generation and storage leasing",
        statement="income_stmt",
        known_labels=[
            "Energy generation and storage leasing", "energy leasing", "energy leasing revenue",
        ],
    ),
    CanonicalMetric(
        metric_id="METRIC_TOTAL_ENERGY_REVENUE",
        display_name="Energy generation and storage segment revenue",
        statement="income_stmt",
        known_labels=[
            "Energy generation and storage segment revenue", "energy segment revenue",
            "total energy revenue", "energy generation and storage revenue",
        ],
    ),
]

# Real 10-K tables use the bare label "Basic" (and "Diluted") for two
# genuinely different concepts under different section headers: a
# per-share dollar figure (units=per_share) under "Net income per share",
# and a share count (units=shares_millions or similar) under "Weighted
# average shares". Confirmed against Tesla's actual FY2025 10-K: both rows
# literally say just "Basic", distinguished only by which table/units they
# belong to. Resolving this by label text alone would be exactly the kind
# of near-miss collision Part 1 of the write-up warns embeddings cause --
# except here it would be the canonicalizer causing it, on an exact-match
# path that is supposed to be the guard against that failure mode. So
# these two labels are excluded from the plain known_labels lists above
# and resolved here instead, using units as the disambiguating signal.
UNITS_DISAMBIGUATED_LABELS: Dict[str, Dict[str, str]] = {
    "basic": {
        "per_share": "METRIC_EPS_BASIC",
        "shares_millions": "METRIC_WEIGHTED_AVG_SHARES_BASIC",
        "shares_thousands": "METRIC_WEIGHTED_AVG_SHARES_BASIC",
        "shares": "METRIC_WEIGHTED_AVG_SHARES_BASIC",
    },
    "diluted": {
        "per_share": "METRIC_EPS_DILUTED",
        "shares_millions": "METRIC_WEIGHTED_AVG_SHARES_DILUTED",
        "shares_thousands": "METRIC_WEIGHTED_AVG_SHARES_DILUTED",
        "shares": "METRIC_WEIGHTED_AVG_SHARES_DILUTED",
    },
}


# =============================================================================
# Label normalization and registry index
# =============================================================================

def _normalize_label(label: str) -> str:
    """Normalize a raw label for exact-match comparison.

    Lowercases, converts curly quotes to straight (real filings mix both,
    e.g. "Stockholders’ equity"), strips trailing footnote markers
    like " (1)", and collapses whitespace. Does not strip other
    parenthetical content: "(loss)" in "Net income (loss)" is meaningful
    and is matched via an explicit known_labels entry, not stripped away.

    Args:
        label: Raw label text as extracted from a filing.

    Returns:
        Normalized label, suitable as a registry lookup key.
    """
    text = label.strip().lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(":").strip()


def _build_label_index(metrics: List[CanonicalMetric]) -> Dict[str, str]:
    """Build the normalized-label to metric_id lookup table.

    Args:
        metrics: The canonical metric registry.

    Returns:
        Mapping from normalized label to metric_id.

    Raises:
        ValueError: If two different metrics claim the same normalized
            label. This is a registry authoring bug and must fail at load
            time rather than silently picking one metric over the other.
    """
    index: Dict[str, str] = {}
    for metric in metrics:
        for raw_label in metric.known_labels:
            key = _normalize_label(raw_label)
            if key in index and index[key] != metric.metric_id:
                raise ValueError(
                    f"canonical registry collision: {key!r} claimed by both "
                    f"{index[key]} and {metric.metric_id}"
                )
            index[key] = metric.metric_id
    return index


_LABEL_INDEX = _build_label_index(CANONICAL_METRICS)


# =============================================================================
# Public entry points
# =============================================================================

class CanonicalizationResult(BaseModel):
    """Result of resolving one raw label against the canonical registry."""
    metric_canonical_id: Optional[str] = None
    ambiguity_flags: List[str] = Field(default_factory=list)


def canonicalize_label(raw_label: str, units: Optional[str] = None) -> CanonicalizationResult:
    """Resolve a raw filing label to a canonical metric id.

    Args:
        raw_label: Label exactly as extracted from the filing, e.g.
            "Total revenues".
        units: The fact's units (e.g. "per_share", "shares_millions"),
            when known. Required to resolve a small set of labels ("Basic",
            "Diluted") that are genuinely ambiguous by text alone -- see
            UNITS_DISAMBIGUATED_LABELS. Not needed for any other label.

    Returns:
        The canonical id if the normalized label is an exact registry
        match (using units to disambiguate where needed), otherwise no id
        and an ambiguity flag explaining why.
    """
    key = _normalize_label(raw_label)

    if key in UNITS_DISAMBIGUATED_LABELS:
        by_units = UNITS_DISAMBIGUATED_LABELS[key]
        if units in by_units:
            return CanonicalizationResult(metric_canonical_id=by_units[units])
        return CanonicalizationResult(
            ambiguity_flags=[
                f"ambiguous_without_units: {raw_label!r} could mean any of "
                f"{sorted(set(by_units.values()))} depending on units; got units={units!r}"
            ],
        )

    metric_id = _LABEL_INDEX.get(key)
    if metric_id is not None:
        return CanonicalizationResult(metric_canonical_id=metric_id)
    return CanonicalizationResult(
        ambiguity_flags=[
            f"unrecognized_label: {raw_label!r} did not match the canonical metric registry"
        ],
    )


def normalize_label(label: str) -> str:
    """Public entry point for the same exact-match-after-normalization rule
    the canonical registry itself uses (lowercase, curly-quote, footnote-
    marker, whitespace normalization). Exposed so a caller matching
    directly against facts' own metric_raw_label -- e.g. the raw-label
    fallback in agents/fact_retriever.py for a metric with no canonical
    registry entry, such as a business-segment breakdown -- uses the exact
    same normalization rule rather than a second, drifting definition of
    "the same label"."""
    return _normalize_label(label)


def all_metric_ids() -> List[str]:
    """Every canonical metric id in the registry, for question types (e.g.
    ranking) that scan across all metrics rather than resolving one."""
    return [m.metric_id for m in CANONICAL_METRICS]


def expense_metric_ids() -> List[str]:
    """Canonical metric ids curated as cost/expense line items. See
    CanonicalMetric.is_expense."""
    return [m.metric_id for m in CANONICAL_METRICS if m.is_expense]


def resolve_canonical_metric(raw_label: str, units: Optional[str] = None) -> Optional[str]:
    """Resolve a raw label to a canonical metric id, or None if unresolved.

    Thin wrapper around canonicalize_label() matching the signature
    agents/fact_retriever.py already calls on the online query path. Never
    makes an LLM call or network request; must stay safe to call from a
    stage declared LLM-free. units is optional and only needed for the
    handful of labels units alone can disambiguate; a user's natural-
    language question ("Tesla's diluted EPS") supplies enough context in
    the phrase itself that units are not needed on this path in practice.

    Args:
        raw_label: Natural-language metric text, e.g. from a user question
            or an extracted fact's raw label.
        units: The fact's units, if known.

    Returns:
        The canonical metric id, or None if unresolved.
    """
    return canonicalize_label(raw_label, units).metric_canonical_id


def canonicalize_facts(
    facts: List[ExtractedFact],
    company_ticker: str,
) -> List[FactRecord]:
    """Canonicalize a batch of extracted facts into insertable records.

    Args:
        facts: Facts produced by ingest/fact_extractor.py.
        company_ticker: Ticker the facts belong to, e.g. "TSLA".

    Returns:
        One FactRecord per input fact, ready for store.db.insert_fact().
        Unresolved facts are not dropped: they carry
        metric_canonical_id=UNRESOLVED_METRIC_ID and an ambiguity_flag, so
        they are stored and visible rather than silently lost.
    """
    records: List[FactRecord] = []
    unresolved_count = 0
    for fact in facts:
        result = canonicalize_label(fact.metric_raw_label, fact.units)
        if result.metric_canonical_id is None:
            unresolved_count += 1
        records.append(FactRecord(
            company_ticker=company_ticker,
            metric_canonical_id=result.metric_canonical_id or UNRESOLVED_METRIC_ID,
            ambiguity_flags=result.ambiguity_flags,
            **fact.model_dump(),
        ))

    log.info(
        "Canonicalized %d facts for %s: %d resolved, %d unresolved",
        len(facts), company_ticker, len(facts) - unresolved_count, unresolved_count,
    )
    return records


# =============================================================================
# CLI entry point (manual testing / full offline pipeline dry run)
# =============================================================================

if __name__ == "__main__":
    import argparse

    from app.ingest.fact_extractor import FilingContext, extract_facts_from_tables
    from app.ingest.pdf_parser import parse_filing
    from app.ingest.table_classifier import classify_tables

    ap = argparse.ArgumentParser(
        description="Run the full offline pipeline: parse, classify, extract, canonicalize.",
    )
    ap.add_argument("pdf_path", help="Path to filing PDF")
    ap.add_argument(
        "--filing-id",
        required=True,
        help="Stable identifier, e.g. TSLA-10K-2025-12-31",
    )
    ap.add_argument("--company-ticker", required=True, help="e.g. TSLA")
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
    extracted = extract_facts_from_tables(parsed.tables, classifications, filing_context)
    canonicalized = canonicalize_facts(extracted, args.company_ticker)
    print(json.dumps([r.model_dump() for r in canonicalized], indent=2))
