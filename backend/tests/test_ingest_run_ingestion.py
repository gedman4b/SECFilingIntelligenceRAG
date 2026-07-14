"""Tests for ingest/run_ingestion.py's prose chunking. Deterministic, no
LLM, no network -- this is what makes narrative retrieval actually see
the text it's supposed to search over."""

from __future__ import annotations

from app.ingest.pdf_parser import ProseSection
from app.ingest.run_ingestion import CHUNK_TARGET_CHARS, _chunk_prose_section


def _section(text: str, section_type: str = "risk_factors") -> ProseSection:
    return ProseSection(
        filing_id="X", section_id="X::section::risk_factors",
        section_type=section_type, heading="Item 1A. Risk Factors",
        page_start=1, page_end=10, text=text,
    )


def test_chunks_stay_near_target_size_for_normal_paragraphs():
    text = "\n\n".join(f"Paragraph {i} about a risk factor." * 5 for i in range(50))
    chunks = _chunk_prose_section(_section(text))
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_TARGET_CHARS * 2 for c in chunks)


def test_falls_back_to_line_splitting_when_whole_section_has_no_blank_lines():
    """The original, already-covered case: zero or one blank-line
    paragraph in the whole section."""
    text = "\n".join(f"Line {i} of unbroken risk factor prose." for i in range(200))
    chunks = _chunk_prose_section(_section(text))
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_TARGET_CHARS * 2 for c in chunks)


def test_falls_back_per_paragraph_when_one_paragraph_is_wildly_oversized():
    """Regression guard for a real bug found via live testing: Tesla's
    actual risk_factors section split on blank lines into exactly 3
    'paragraphs' (not <=1, so the old code never fell back), but two of
    them were 60,000+ characters -- effectively the whole section minus a
    couple of accidental blank-line breaks, embedded as one indivisible
    chunk. An embedding model only encodes roughly its first ~1500-2500
    characters, so the other 98%+ of that content was invisible to every
    future search. A short normal paragraph, one enormous unbroken
    paragraph, and another short one must produce chunks that are all
    close to CHUNK_TARGET_CHARS, not one enormous chunk."""
    normal_para = "A short risk paragraph about competition. " * 3
    huge_para = "\n".join(f"Risk line {i} about supply chain constraints and tariffs." for i in range(2000))
    text = f"{normal_para}\n\n{huge_para}\n\nAnother short closing paragraph."
    chunks = _chunk_prose_section(_section(text))
    assert len(chunks) > 5  # the huge paragraph alone must have been split into many
    assert all(len(c.text) <= CHUNK_TARGET_CHARS * 2 for c in chunks), \
        f"chunk sizes: {[len(c.text) for c in chunks]}"


def test_moderately_long_single_paragraph_is_not_needlessly_resplit():
    """A genuinely long-but-real paragraph (a bit over target, nowhere
    near the 3x-oversized threshold that signals unbroken raw text) is
    left as a coherent unit rather than being fragmented mid-sentence."""
    text = "This is one real, if verbose, paragraph about risk. " * 40  # ~2100 chars
    assert CHUNK_TARGET_CHARS < len(text) < CHUNK_TARGET_CHARS * 3
    chunks = _chunk_prose_section(_section(text))
    assert len(chunks) == 1
    assert chunks[0].text.strip().startswith("This is one real")


def test_chunk_ids_are_sequential_and_carry_section_metadata():
    text = "\n\n".join(f"Paragraph {i}." * 100 for i in range(10))
    chunks = _chunk_prose_section(_section(text))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"X::section::risk_factors::chunk{c.chunk_index}" for c in chunks)
    assert all(c.section_type == "risk_factors" and c.filing_id == "X" for c in chunks)
