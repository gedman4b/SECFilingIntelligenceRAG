"""Tests for main.py's citation-URL resolution. Deterministic, no LLM,
no network -- this is what turns a stored /pdfs/<filename> path into a
link a real user's browser can actually open, regardless of whether this
backend is answering from localhost, a Vercel preview, or production."""

from __future__ import annotations

from app.main import _absolutize_filing_urls
from app.schemas import Fact, Period


def _fact(filing_url: str) -> Fact:
    return Fact(
        value=1.0, units="USD_millions", period=Period(year=2025),
        metric_canonical_id="METRIC_TOTAL_REVENUE", metric_raw_label="Total revenues",
        is_gaap=True, filing_id="X", filing_url=filing_url, page_number=1,
        table_id="X::p1::t1", row_id=1,
    )


def test_relative_pdf_path_becomes_absolute_against_request_host():
    facts = [_fact("/pdfs/tsla-10-K%2020251231.pdf")]
    _absolutize_filing_urls(facts, "http://localhost:8000/")
    assert facts[0].filing_url == "http://localhost:8000/pdfs/tsla-10-K%2020251231.pdf"


def test_works_against_a_deployed_https_host():
    facts = [_fact("/pdfs/aapl-10-K%2020250927.pdf")]
    _absolutize_filing_urls(facts, "https://secfilintbackend.vercel.app/")
    assert facts[0].filing_url == "https://secfilintbackend.vercel.app/pdfs/aapl-10-K%2020250927.pdf"


def test_mutates_every_fact_in_the_list_in_place():
    facts = [_fact("/pdfs/a.pdf"), _fact("/pdfs/b.pdf")]
    _absolutize_filing_urls(facts, "http://localhost:8000/")
    assert facts[0].filing_url == "http://localhost:8000/pdfs/a.pdf"
    assert facts[1].filing_url == "http://localhost:8000/pdfs/b.pdf"


def test_leaves_an_already_absolute_url_untouched():
    """Defensive: a fact whose filing_url somehow isn't the stored
    /pdfs/... form (e.g. test fixtures elsewhere in the suite use
    arbitrary values like "file:///tsla.pdf") must not be mangled by
    urljoin, which would silently produce a nonsense URL for anything
    that doesn't start with '/'."""
    facts = [_fact("file:///tsla.pdf")]
    _absolutize_filing_urls(facts, "http://localhost:8000/")
    assert facts[0].filing_url == "file:///tsla.pdf"
