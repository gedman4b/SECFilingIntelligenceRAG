# SEC Filing Intelligence — Requirements Traceability

This document maps every requirement in the original assignment brief
(`SEC_Filing_Intelligence_Take-Home_v2_1.docx`, the Deloitte take-home
prompt) to what is actually built, with file/line references and
verification evidence. It is deliberately separate from
`backend/app/docs/Scott_Josephson_Deloitte_SEC_Filing_Intelligence_TakeHome.docx`,
which is the architectural-reasoning narrative for the interview
walkthrough — this document is the checklist proving that narrative is
grounded in real, tested, running code, not just design intent.

Live deployment: backend `https://secfilintbackend.vercel.app`, frontend
`https://secfilingsint.vercel.app`. Both confirmed live as of this
writing (`GET /health` → 200 on the backend; frontend serving 200).

Test suite: 202 pytest tests, all passing, confirmed hermetic (the full
suite passes with `ANTHROPIC_API_KEY` explicitly unset — no test depends
on real network access, including the new LLM-assisted resolution tier
in §6, per AGENTS.md: "No test that requires network access to a live
LLM provider runs in the default suite"). Eval benchmark: 20/20 cases
passing (`app/eval/benchmark.py` / `app/eval/runner.py`).

---

## 1. Data Environment: PDF as primary source, not XBRL

> "Your system should treat PDF filings as the primary data source, not
> the clean, pre-tagged XBRL JSON that the EDGAR API would otherwise hand
> you for free."

**Built:** The entire pipeline reads real PDFs via `pdfplumber`
(`app/ingest/pdf_parser.py`). Nowhere in the codebase is there an XBRL
parser, an EDGAR API client, or any dependency on pre-tagged structured
data. The corpus is 6 real SEC filings, not synthetic fixtures:

| Filing ID | Company | Form | Period |
|---|---|---|---|
| `AAPL-10K-2025-09-27` | Apple | 10-K | FY2025 |
| `AAPL-10Q-2025-12-27` | Apple | 10-Q | Q1 FY2026 |
| `AAPL-10Q-2026-03-28` | Apple | 10-Q | Q2 FY2026 |
| `TSLA-10K-2025-12-31` | Tesla | 10-K | FY2025 |
| `TSLA-10KA-2025-12-31` | Tesla | 10-K/A | FY2025 (amended) |
| `TSLA-10Q-2026-03-31` | Tesla | 10-Q | Q1 FY2026 |

The brief's own "Suggested Scope" explicitly sanctions "3–5 filings for
1–2 companies is sufficient" — 6 filings across 2 companies, including a
genuine amended filing (10-K/A), matches that guidance directly.

**Offline vs. online split**, per the brief's own framing question ("does
it make more sense to parse the entire corpus once, up front, or to
retrieve and interpret raw PDF content live, per question?"): parsing,
table classification, fact extraction, and canonicalization all happen
once, offline, in `app/ingest/run_ingestion.py`. The online `/query` path
(`app/main.py`) never touches a PDF for numeric questions — it queries a
pre-structured SQLite fact store. This is the single biggest architectural
decision in the system and is argued explicitly in Part 1 of the write-up
docx ("The offline pipeline pays the interpretation cost").

---

## 2. Functional Expectation 1 — Answer natural-language financial questions

**Built:** Six question types, each a distinct, deterministic-once-planned
path:

| `QuestionType` | Handles | Retrieval | Arithmetic |
|---|---|---|---|
| `numeric_lookup` | single value | `fact_retriever.retrieve_facts()` | none |
| `growth_calc` | one metric, two periods | same | `numerical_reasoner.compute()` |
| `comparison` | one metric, raw delta | same | same |
| `margin_calc` | ratio of two metrics | `retrieve_ratio_facts()` | same |
| `ranking` | cross-metric magnitude ranking | `retrieve_ranking_facts()` | same |
| `narrative` | qualitative/MD&A/risk content | `prose_retriever.retrieve_prose()` | none (no numbers involved) |

(`app/schemas.py:10` — `QuestionType` enum; `app/main.py` — routes each
type to the right stage.)

**Evidence:** All 6 types are live-tested against the real deployed
backend during this project (not just unit-mocked), including edge cases
like margin questions needing multi-period deltas and ranking questions
needing date-grounded period resolution. `app/eval/benchmark.py` covers
`numeric_lookup` (10 cases), `growth_calc` (6), `comparison` (2),
`narrative` (2) with hand-verified ground truth. `margin_calc` and
`ranking` are validated by live testing with hand-checked arithmetic
rather than the automated benchmark — a named, tracked gap, not a silent
one (see §11).

---

## 3. Functional Expectation 2 — Show traceability

> "filing source, reporting period, table/section references, extracted
> values, calculation steps, citations, or intermediate reasoning
> artifacts."

**Built:**
- Every `Fact` (`app/schemas.py`) carries `filing_url`, `page_number`,
  `table_id`, `row_id`, `is_gaap`, `is_restated`, `is_audited`,
  `is_preferred_source` — full provenance, not just a value.
- `QueryResponse.citations` is a structured list built directly from
  `Fact` objects (`app/agents/answer_composer.py`), never LLM-generated
  text.
- `QueryResponse.computation_expression` shows the literal arithmetic
  string (e.g. `"(94,827 - 97,690) / |97,690| * 100 = -2.93%"`), and the
  Composer is instructed to quote it verbatim, never restate it in its
  own words.
- **Citation links actually resolve.** This required real engineering:
  the backend now serves the source PDFs directly (`app/main.py:73-75`,
  `StaticFiles` mount at `/pdfs`), and `filing_url` is stored as a
  host-relative path at ingestion time (`app/ingest/run_ingestion.py`)
  and resolved to an absolute URL per-request using the actual incoming
  request's host (`_absolutize_filing_urls()`, `app/main.py`). This
  replaced an earlier version that stored `file:///home/.../pdfs/...`
  paths — meaningless to anyone except the machine ingestion ran on.
  Verified live: fetching a citation URL returns the real PDF,
  byte-for-byte identical to the source file.
- Frontend (`frontend/public/app.js`, `index.html`) surfaces all of this:
  a confidence badge, the calculation block, a warnings panel
  (severity-colored), and an expandable sources table with clickable
  citation links.

---

## 4. Functional Expectation 3 — Handle ambiguity honestly

> "missing data, inconsistent labels, multiple possible interpretations,
> quarterly vs. annual confusion, unaudited statements, amended filings,
> non-GAAP metrics, or confidence limitations."

Each item in the brief's own checklist, mapped to a specific, named
Verifier check (`app/agents/verifier.py`) — 13 checks total, each tied to
one concrete failure mode, not a generic "confidence score":

| Brief's ambiguity type | Check | Mechanism |
|---|---|---|
| Missing data | Check 2 | empty `facts` → `insufficient_data` |
| Missing company (a *deterministic* guard, not LLM judgment) | Check 1b | fails closed if `company_ticker` unset, regardless of whether the Planner happened to flag it |
| Inconsistent / unrecognized labels | Check 6 | `ambiguity_flags` on a `Fact` → confidence downgraded to `medium` |
| Quarterly vs. annual confusion | schema-level | `Period.is_ytd`, `Period.quarter` prevent a YTD or quarterly figure silently substituting for an annual one |
| Unaudited statements | Check 10 | `is_audited` derived deterministically from `form_type` (10-Q ⇒ unaudited), disclosed as info |
| Amended filings | Check 5 | `is_restated` disclosed; corpus contains a real 10-K/A |
| Non-GAAP metrics | Check 4 | `gaap_preference` + `is_gaap`, downgrades confidence on mismatch |
| Confidence limitations | schema-level | `confidence: high \| medium \| low \| insufficient_data`, and the Composer is forbidden from ever restating it in different words than the exact given value |
| Multi-source conflicts (not explicitly named in the brief, but a real corpus phenomenon) | Check 9 | same (metric, period) reported differently by 2+ tables → disclosed if resolvable, fails closed if not |
| Margin/ranking-specific insufficiency | Checks 2b, 2c, 3, 7, 8 | question-type-specific fail-closed conditions |
| Metric synonym gaps (a phrasing gap, not named in the brief's list verbatim, but the same "multiple possible interpretations" concern) | Check 6 | a match found only via the LLM-assisted synonym tier (§6) is flagged `llm_synonym_match` on the `Fact`, downgrading confidence to `medium` — never silently treated as an exact match |

**A concrete, hard-won example of "fail closed, don't guess":** Tesla's
own income statement reuses the *exact text* "Automotive sales" for two
different line items — once under "Revenues" ($15,473M for Q1 2026) and
again, verbatim, under "Cost of revenues" ($12,616M, same quarter). An
earlier attempt to add this as a canonical metric silently merged the two
values. Caught via live testing, reverted, and documented in
`app/ingest/canonicalizer.py` with the full reasoning — the system now
correctly reports a "conflicting values, no single preferred source"
error for this label rather than guessing, and the fix required a real
architectural decision (leave it unresolved until extraction captures
which table section a row came from) rather than a quick patch.

---

## 5. Functional Expectation 4 — Demonstrate an agentic/workflow-oriented approach

**Built:** Six-agent pipeline, each a separate module with a single
responsibility and a typed Pydantic contract at every boundary
(`app/schemas.py`):

```
Query Planner (LLM, tool-use schema-constrained)
   -> Fact Retriever / Numerical Reasoner   (numeric-family types)
   -> Prose Retriever                        (narrative)
   -> Verifier (13 deterministic checks, the trust boundary)
   -> Answer Composer (LLM, narrates only pre-verified facts)
```

- **Planning:** `app/agents/planner.py` — LLM classifies the question
  into a structured `QueryPlan` via Anthropic tool-use (the schema is
  enforced by the API itself, not by hoping a prompted JSON blob parses).
- **Retrieval / reasoning split:** `app/agents/fact_retriever.py` (pure
  SQL, zero LLM calls) and `app/agents/numerical_reasoner.py` (pure
  Python arithmetic, zero LLM calls) are separate modules on purpose —
  retrieval and computation are different failure modes and are hardened
  independently.
- **Verification pass:** `app/agents/verifier.py` — gates every response
  before composition; explicitly named in the brief's own list of
  agentic patterns ("verification passes").
- **Reconciliation logic:** `resolve_preferred_facts()`
  (`app/agents/fact_retriever.py`) picks a winner among multiple
  corroborating source tables by table richness; Check 9 in the Verifier
  handles the case where reconciliation *can't* pick a winner.
- **Fallback mechanisms:** metric resolution is a three-tier cascade
  (`_resolve_metric_facts()`, `app/agents/fact_retriever.py:211`): the
  raw-label fallback (`_retrieve_facts_by_raw_label()`,
  `app/agents/fact_retriever.py:111`) for metrics with no canonical
  registry entry, then LLM-assisted synonym resolution
  (`resolve_canonical_metric_via_llm()`,
  `app/ingest/canonicalizer.py:561`, detailed in §6) as the last resort
  before failing closed; fail-closed `insufficient_data` as the universal
  fallback when any stage still can't produce a trustworthy answer.

This is not incidental structure — it is the direct response to the
brief's own diagnosis of why the *previous* system failed ("it
hallucinated values... mixed annual and quarterly numbers... confused
GAAP versus non-GAAP... provided answers without traceability"). Each of
those four failure modes maps to a specific stage or check above, not a
prompting fix.

---

## 6. Functional Expectation 5 — Justify retrieval/extraction for numeric tables

This is the section the brief calls out as most heavily weighted
("we're calling it out as its own expectation rather than leaving it
implicit").

**Retrieval mechanism and why:** Numeric lookups are deterministic SQL
(`WHERE company_ticker = ? AND metric_canonical_id = ? AND year = ? AND
quarter = ?`), never embeddings. Prose (narrative) retrieval uses
Chroma + a local embedding model. This split is Guard 1 of the design:
*"numeric queries never use embeddings."* Metric resolution — turning a
user's phrase into a `metric_canonical_id` — is a three-tier cascade
(`agents/fact_retriever.py:_resolve_metric_facts()`), cheapest and
safest first:
1. **Exact canonical match** (`resolve_canonical_metric()`) — free,
   deterministic, the common case.
2. **Raw-label fallback** (`_retrieve_facts_by_raw_label()`) — still
   exact-match, just against facts' own stored raw label text instead of
   the curated registry, for a metric with no registry entry at all
   (e.g. a business-segment line).
3. **LLM-assisted synonym resolution** (`resolve_canonical_metric_via_llm()`,
   `app/ingest/canonicalizer.py`) — the last resort, only reached when
   both tiers above find nothing.

**What embeddings are good/bad at, and how that shaped the design —
including tier 3:** Embeddings encode topical similarity, not exact
label identity or numerical magnitude — "Net income" and "Net income
attributable to common stockholders" would embed as nearly identical
vectors despite being different dollar figures. The registry
(`app/ingest/canonicalizer.py`) resolves raw labels to canonical IDs by
exact-match-after-normalization specifically to prevent this collision.
This was validated the hard way, three times, this project:
- The Net income / Net income attributable near-miss the brief itself
  names is a real, distinct canonical ID pair in the registry.
- The "Automotive sales" revenue-vs-cost-of-revenue collision (§4 above)
  is the *same class of bug*, discovered live, and fixed by refusing to
  force a canonical mapping rather than by loosening the matching rule.
- Tier 3 itself exists because of a third instance of the identical
  underlying problem, seen from the opposite direction: real, answerable
  questions ("Tesla's profit", "revenue from car sales") were failing
  closed for want of a curated synonym, and manually patching the
  registry one hand-noticed live-test failure at a time doesn't scale.

Tier 3 is deliberately **not** the thing Guard 1 bans, even though it is
also LLM-assisted. Guard 1's actual concern is a *continuous* similarity
search that always returns its nearest neighbor, ranked by distance, with
no way to say "none of these" — exactly the near-miss-collision failure
mode above. Tier 3 instead does closed-set *classification*: the model
picks from the exact, already-curated list of registered metric ids (the
same 22-entry registry, not a separate embedding index) or declines, and
its answer is validated against that literal list before being trusted at
all — a hallucinated id is treated identically to an explicit decline,
fail closed. It is closer in kind to what the Query Planner already does
(LLM extracts structured intent from open-ended text, validated by
Pydantic afterward) than to embedding-based nearest-neighbor search. Every
match found this way is flagged `llm_synonym_match` on the resulting
`Fact` and downgrades confidence to `medium` (Check 6) — it can never
look as certain as an exact match, so the honest-uncertainty guarantee
holds even though the coverage gap is closed. Verified live, not just
unit-tested: "How much cash does Tesla have on hand?" — a phrase that
appears nowhere in the code or tests — correctly resolved to
`METRIC_CASH_AND_EQUIVALENTS` with the real value and an explicit
`confidence: medium`; separately, "margin" alone (genuinely ambiguous
between gross/operating/net margin) was correctly **declined** rather
than guessed.

**Where arithmetic happens:** 100% in `app/agents/numerical_reasoner.py`.
No other module performs arithmetic on a `Fact.value`. This was not just
asserted — it was *broken twice by live testing* and closed both times by
moving more computation into this module, not by adding another prompt
instruction:
1. A margin question needing two periods' ratios initially only computed
   the most recent period; the Composer "helpfully" computed the older
   period's ratio itself. Fixed: every requested period's ratio is now
   computed here.
2. A trend question ("how much did SG&A grow as a % of revenue") needing
   a percentage-point *delta* between two already-computed ratios hit the
   same failure one level up — the Composer subtracted them itself. Fixed:
   the delta is now computed here too.

Both fixes are documented as regression tests
(`backend/tests/test_agents_numerical_reasoner.py`,
`test_margin_calc_computes_every_requested_period_not_just_latest`,
`test_margin_calc_includes_precomputed_delta_between_periods`) and as a
concrete anecdote in Part 8 of the write-up docx.

**How correctness is benchmarked:** `app/eval/benchmark.py` — 20
hand-verified question/answer pairs with a specific source page and
expected value for each, run against the deterministic pipeline directly
(bypassing the LLM Planner/Composer) via `app/eval/runner.py`, which
exits non-zero on any numeric failure — safe to use as a CI gate.
"Correct" is defined on value + period + provenance, not just a matching
number (documented in the write-up docx Part 7).

---

## 7. Suggested Scope

The brief explicitly endorses scope reduction ("you may choose to
support only: one company... a subset of metrics... structured numeric
questions only... strong candidates usually make thoughtful cuts rather
than attempting everything"). Actual scope:

- **2 companies** (Tesla, Apple), not the full archive.
- **6 filings**, not "tens of thousands."
- **22 canonical metrics** (`app/ingest/canonicalizer.py`), covering
  consolidated income statement, balance sheet, cash flow, EPS/share
  count, and Tesla's automotive/energy segment revenue lines — not
  hundreds of line items or non-GAAP reconciliations.
- **Live EDGAR API**: not used at all in the query path (per the brief's
  own guidance to treat it as a constrained, low-volume resource). The
  corpus was sourced as local PDFs and ingested once, offline.

---

## 8. Example Questions — mapped to what's supported

The brief's example list names companies not in this prototype's 2-company
corpus (Microsoft, Amazon, NVIDIA, Google, Meta, Netflix) — an explicitly
sanctioned scope cut, not a gap. What's verified is that every *question
pattern* those examples represent is implemented and either
benchmark-tested or live-tested with hand-checked arithmetic:

| Brief's example (paraphrased) | Pattern | Status |
|---|---|---|
| "net income growth between 2025 and 2026" | `growth_calc` | ✅ benchmark-tested |
| "Q1 revenue change year-over-year" | `growth_calc`, quarterly | ✅ benchmark-tested |
| "total revenue in [year]" | `numeric_lookup` | ✅ benchmark-tested |
| "net income in the latest annual filing" | `numeric_lookup` | ✅ benchmark-tested |
| "AWS revenue last quarter" (segment metric, no registry entry) | raw-label fallback | ✅ live-tested equivalent: Tesla's "Energy generation and storage segment revenue" |
| "operating income for Q1 2026" | `numeric_lookup` | ✅ benchmark-tested |
| "cash and cash equivalents" | `numeric_lookup` | ✅ benchmark-tested |
| "diluted EPS in the latest filing" | `numeric_lookup` | ✅ supported (`METRIC_EPS_DILUTED`) |
| "operating cash flow change" | `comparison` | ✅ benchmark-tested |
| "biggest expense increases" | `ranking`, `ranking_scope="expense"` | ✅ live-tested: Tesla expense ranking, correct math verified by hand |
| "SG&A grow as a percentage of revenue" | `margin_calc` | ✅ live-tested: Tesla SG&A/revenue, both periods + delta verified by hand |
| "gross margin for the last two years" | `margin_calc`, multi-period | ✅ live-tested: Tesla gross margin FY2023–FY2025, verified by hand |
| "which metrics deteriorated the most" | `ranking`, `most_deteriorated` | ✅ live-tested, including the deterministic no-company-specified guard (Check 1b) |
| "what did management cite as risks" | `narrative` | ✅ live-tested; also where a real prose-chunking bug was found and fixed this session (see §11) |
| "net income growth" phrased colloquially, e.g. "change in Tesla's profit" (a real user-reported failure: bare "profit" wasn't a curated synonym) | `growth_calc`, exact-match tier | ✅ live-tested: fixed by adding "profit" as a curated `METRIC_NET_INCOME` synonym after confirming no collision risk |
| a metric phrase with no curated synonym at all, e.g. "cash on hand" | `numeric_lookup`, LLM-assisted tier (§6) | ✅ live-tested: resolved correctly via tier 3, `confidence: medium`, `llm_synonym_match` disclosed |

---

## 9. Deliverables checklist

1. **A tangible artifact** — ✅ not a mockup: a working FastAPI backend and
   Node.js frontend, both deployed and live (`secfilintbackend.vercel.app`,
   `secfilingsint.vercel.app`), backed by a real 6-filing corpus and a
   real Anthropic API integration, not stubbed responses.
2. **A 5–10 minute walkthrough** — supported by the write-up docx's Part 2
   architecture diagram (both the ASCII version and a regenerated visual
   diagram reflecting all 6 question types) and this document's mapping.
3. **Written/verbal discussion** (Trust & hallucination, Agentic workflow
   design, Product thinking) — the write-up docx Part 8 covers all three
   explicitly, including two *real* live-tested hallucination bugs (not
   hypothetical ones) and how each was closed architecturally.

---

## 10. Self-assessment against "What We Are Evaluating"

- **Applied reasoning / engineering judgment:** every fix in this
  project's history started from a live-tested failure, not a
  speculative one — e.g. the margin_calc arithmetic-leakage bugs, the
  Automotive sales collision, the prose-chunking bug (§11) were all
  found by actually running questions against the deployed system, not
  inferred from reading the code. The LLM-assisted synonym tier (§6) is
  the same discipline applied to a design *decision*, not just a bug: it
  was built only after confirming, precisely, why the obvious-looking
  fix ("just use the vector database for this too") would have
  reintroduced the exact near-miss-collision risk this project exists to
  prevent, and designed around that constraint instead of past it.
- **Trust & traceability:** the confidence field is never allowed to
  contradict the narrative text (Composer rule 7); citation links
  resolve to real, byte-identical PDFs; the Verifier is a hard gate, not
  a soft heuristic — `insufficient_data` responses never reach the
  Composer's LLM call at all (`answer_composer.py`: `if confidence ==
  'insufficient_data': ... return ... # no LLM call`).
- **Communication:** this document plus the write-up docx together are
  meant to make the *why* legible, not just the *what*.
- **AI leverage vs. fundamentals:** Claude Code was used throughout
  implementation, but every fix in this document was verified against
  real behavior (tests + live HTTP calls against the actual deployed
  system), not accepted on the model's say-so — the repeated pattern of
  "live-tested, found a real bug, fixed it, added a regression test" is
  the operating discipline, not an afterthought.

---

## 11. Known limitations (honest, not hidden)

- `margin_calc` and `ranking` are not yet in the automated
  `eval/benchmark.py` golden set — validated only by live testing so far.
  A regression in either would not be automatically caught the way a
  `growth_calc` regression would be.
- A table-of-contents page is glued onto the front of the MD&A section
  during PDF parsing for at least one filing (likely because the TOC page
  itself lists "Item 7. Management's Discussion and Analysis..." and
  confuses section-boundary detection). This still occasionally
  outranks real content in narrative retrieval. Fixing it means touching
  `pdf_parser.py`'s section-splitting logic — deferred as higher-risk
  than the prose-chunking bug that was fixed (see below), since it
  affects every section boundary, not one symptom.
- "Automotive sales," "Automotive leasing," and "Services and other" are
  deliberately left unresolved (§4) because Tesla's own filing reuses the
  exact label for two different figures. Properly resolving them needs
  the extraction pipeline to capture which table section a row came
  from — a real, scoped follow-up, not implemented.
- The corpus covers 2 companies; the brief's example questions naming
  other companies (Microsoft, Amazon, etc.) will honestly return
  `insufficient_data`, not an error — this is the system correctly
  reporting a real scope limit, not a bug.
- The LLM-assisted synonym tier (§6) adds a real LLM call, with its
  latency and cost, to any query whose metric phrase resolves at neither
  of the two deterministic tiers — including genuinely unanswerable
  questions, which now cost one extra API call before correctly failing
  closed. Its *precision* is tested (every unit test mocks the model's
  response and checks the validation/flagging logic around it), but its
  *recall* — whether the real, live model actually resolves a given novel
  phrase correctly — is only checked by live testing, not by the
  automated suite, since the tests deliberately never call the real API.
  A future prompt change or model upgrade that made it decline more
  often, or resolve things at lower quality, would not be caught by
  `pytest` the way a `growth_calc` regression would be.
- A significant, previously-invisible bug was found and fixed this
  session: `_chunk_prose_section()` (`app/ingest/run_ingestion.py`) was
  supposed to split narrative sections into ~1,500-character chunks, but
  its fallback logic only triggered when an entire section produced ≤1
  blank-line-delimited paragraph. Tesla's ~150,000-character risk_factors
  section split into exactly 3 such "paragraphs," two of which were
  62,585 and 87,226 characters — effectively unbroken raw text. Since an
  embedding model only encodes roughly the first ~1,500–2,500 characters
  of its input, over 98% of that section's actual content was invisible
  to every narrative search, silently, until this was found and fixed.
  Re-chunking the full corpus with the fix took Tesla's 10-K from 27 to
  271 prose chunks. This is disclosed here because it is exactly the kind
  of quiet, structural failure mode a "policy of appearing more certain
  than the system actually is" would leave undiscovered — the fix and
  its discovery process are the more honest story than claiming the
  system got chunking right the first time.
