"""
ingest/canonicalizer.py

Metric canonicalization: deterministic mapping from raw filing labels to a
fixed canonical metric registry.

Design principle
-----------------
Ingestion-time canonicalization (canonicalize_label(), canonicalize_facts())
is fully deterministic and stays that way: exact-match-after-normalization
against a curated registry (Part 3 of the write-up: "Raw labels are mapped
to a canonical metric registry"). A raw filing label that is not an exact
match, after normalizing case, whitespace, curly quotes, and footnote
markers, is left unresolved and flagged rather than guessed (Part 3 Guard
2). Facts with an unresolved metric are still stored, per the write-up, so
the Verifier can surface the uncertainty instead of the fact silently
disappearing. This path is deliberately never LLM-assisted: fact_extractor.py
already used an LLM to read the label off the page, and this stage exists
specifically as the deterministic cross-check on that -- an LLM guess here
too would remove the check rather than add one.

Query-time resolution (resolve_canonical_metric(), called from
agents/fact_retriever.py) is a different situation: a user's natural-
language phrase ("Tesla's profit") frequently isn't the filing's own
label text ("Net income") at all, so exact-match-only resolution left
real, answerable questions failing closed for want of a curated synonym
-- confirmed repeatedly via live testing ("revenue from car sales" vs.
"Automotive sales", "profit" vs. "Net income"). resolve_canonical_metric_via_llm()
is the fix, and it is deliberately NOT the same thing Part 1 / Part 3
Guard 1 bans embeddings from doing. Guard 1's concern is a *continuous*
similarity search that always returns its nearest neighbor, ranked by
distance, with no way to say "none of these" -- exactly how "Net income"
and "Net income attributable to common stockholders" would embed as
near-identical vectors despite being different dollar figures. This
function instead does closed-set *classification*: the model picks
from the exact, already-curated list of metric ids (the same registry
below) or declines, its answer is validated against that list before
being trusted (a hallucinated id is treated as no match), and it only
runs as a last resort after both the exact match and the raw-label
fallback have already failed. It is closer in kind to what the Query
Planner already does (LLM extracts structured intent from open-ended
text, validated by Pydantic afterward) than to embedding-based nearest-
neighbor search. Every match found this way is still flagged as
ambiguous by the caller, downgrading confidence exactly like the
raw-label fallback already does -- this fills a real gap without
silently claiming more certainty than a curated exact match earns.

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

import anthropic
from pydantic import BaseModel, Field

from app.ingest.fact_extractor import ExtractedFact
from app.store.db import FactRecord

log = logging.getLogger(__name__)
_client = anthropic.Anthropic()

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


_SYNONYM_RESOLUTION_SYSTEM_PROMPT = '''You match a natural-language financial term to ONE canonical metric from a fixed list, or to none of them.

Rules:
1. You may ONLY return a metric_id that appears in the list below, or null. Never invent an id that is not listed.
2. Return null if the phrase is genuinely ambiguous between two or more listed metrics, or does not clearly refer to any single one of them. A vague, unqualified term that could plausibly mean several different listed metrics (e.g. "margin" alone, which could be gross, operating, or net margin) must return null, not a guess.
3. A colloquial or informal term should resolve to whichever listed metric it conventionally refers to in everyday financial usage (e.g. "profit" or "the bottom line" conventionally means net income, "the top line" conventionally means revenue) -- but only when that convention is genuinely unambiguous, per rule 2.
4. Never let an unqualified term collapse into a qualified metric, or vice versa. If the list has separate entries for a metric and a more specific qualified version of it (e.g. a plain profit/income metric versus one scoped to "attributable to common stockholders", or versus "operating" or "gross" specifically), an unqualified query must resolve to the unqualified metric and must never match the qualified one, and a qualified query must never match the unqualified metric.
'''

_metric_ids_cache: Optional[set] = None


def _valid_metric_ids() -> set:
    global _metric_ids_cache
    if _metric_ids_cache is None:
        _metric_ids_cache = set(all_metric_ids())
    return _metric_ids_cache


def _build_registry_description() -> str:
    lines = []
    for m in CANONICAL_METRICS:
        examples = ", ".join(repr(l) for l in m.known_labels[:3])
        lines.append(f"- {m.metric_id}: {m.display_name} (e.g. {examples})")
    return "\n".join(lines)


_SYNONYM_TOOL = {
    'name': 'submit_metric_match',
    'description': 'Submit which canonical metric, if any, the phrase refers to.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'metric_id': {
                'type': ['string', 'null'],
                'description': 'One of the listed metric ids, or null if none clearly and unambiguously match.',
            },
        },
        'required': ['metric_id'],
    },
}


def resolve_canonical_metric_via_llm(natural_language_phrase: str) -> Optional[str]:
    """Last-resort, query-time-only fallback when a phrase matched
    nothing in the deterministic registry (resolve_canonical_metric())
    and nothing via the raw-label fallback
    (agents/fact_retriever._retrieve_facts_by_raw_label()).

    See the module docstring for why this is a different, safer
    mechanism than embeddings despite also being "AI-assisted": this is
    closed-set classification over the already-curated registry, not
    open-ended similarity search. The model's answer is validated
    against the real registry before being trusted at all.

    Deliberately not used by canonicalize_label()/canonicalize_facts()
    (the ingestion-time, raw-filing-label path), which stays fully
    deterministic -- see the module docstring.

    Args:
        natural_language_phrase: Text that already failed exact-match
            canonical resolution and the raw-label fallback, e.g.
            "profit" or "revenue from car sales".

    Returns:
        A canonical metric id if the model picked one AND it is a real,
        currently-registered id, otherwise None. A hallucinated id (not
        in the registry) is treated identically to an explicit null --
        fail closed, never trust an unvalidated model output.

    Callers MUST treat a non-None result as an ambiguous match, not an
    exact one -- e.g. by adding an ambiguity_flag to the resulting
    Fact(s) so the Verifier's Check 6 downgrades confidence, the same
    honest-uncertainty treatment the raw-label fallback already gets.
    This function has no access to a Fact object and does not do that
    itself; see agents/fact_retriever.py's caller.
    """
    system_prompt = (
        f"{_SYNONYM_RESOLUTION_SYSTEM_PROMPT}\n\nCanonical metrics:\n{_build_registry_description()}"
    )
    try:
        response = _client.messages.create(
            model='claude-sonnet-4-5',
            max_tokens=200,
            system=system_prompt,
            tools=[_SYNONYM_TOOL],
            tool_choice={'type': 'tool', 'name': 'submit_metric_match'},
            messages=[{'role': 'user', 'content': natural_language_phrase}],
        )
        tool_use = next(b for b in response.content if b.type == 'tool_use')
        metric_id = tool_use.input.get('metric_id')
    except Exception as e:
        # Fail closed: no match instead of guessing, same as every other
        # stage in this system when an LLM call errors.
        log.warning("LLM synonym resolution failed for %r: %s", natural_language_phrase, e)
        return None

    if metric_id not in _valid_metric_ids():
        if metric_id is not None:
            log.warning(
                "LLM synonym resolution returned an unrecognized metric_id %r for %r; treating as no match",
                metric_id, natural_language_phrase,
            )
        return None

    log.info("LLM synonym resolution: %r -> %s", natural_language_phrase, metric_id)
    return metric_id


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
