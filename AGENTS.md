# AGENTS.md

Operating instructions for Claude Code (and any other AI coding agent) working
on this project. Read this file at the start of every session.

## Project

SEC Filing Intelligence prototype. FastAPI backend, Node.js frontend, agentic
workflow over PDF SEC filings. Deliverable for the Deloitte Agentic AI
Engineer take-home. The goal is a lightweight but architecturally sound
system that answers natural-language financial questions with full
traceability and honest uncertainty.

## The master spec is the write-up, not this file

The architectural reasoning, agent decomposition, failure-mode-to-mitigation
mapping, and evaluation approach live in:

  `docs/SEC_Filing_Intelligence_TakeHome.docx`

Read it before starting any session. Every architectural decision in this
project is justified there against a specific failure mode of the previous
system at the client. **If this file and the write-up disagree, the write-up
wins.**

## Reference implementations

Three files establish the code style, docstring conventions, schema patterns,
and design principles for the rest of the project. Match their patterns
when producing new modules:

- `backend/app/ingest/pdf_parser.py`
  Python module conventions: Pydantic v2 schemas, module-level design
  principle in the docstring, Google-style function docstrings, structured
  logging, graceful failure handling with warnings collected rather than
  aborting, CLI entry point at bottom for manual testing.

- `frontend/public/styles.css`
  Frontend styling: white background, black foreground, color used only
  for signal (confidence, warnings), no gradients or drop shadows,
  responsive at 640px breakpoint.

- `frontend/package.json`
  Node.js: minimal dependencies, engines pinned to Node 18+, private true,
  license UNLICENSED, no build step.

## Non-negotiable architectural rules

These are not preferences. Violating any of them breaks the trust model
the write-up establishes. If you find yourself wanting to violate one,
stop and surface the proposal to Scott instead of writing code.

1. **The LLM never performs arithmetic.** All growth rates, margins, deltas,
   ratios, and year-over-year calculations happen in
   `agents/numerical_reasoner.py` in deterministic Python. The Answer
   Composer narrates the result but does not compute it.

2. **The Verifier gates every response.** No answer reaches the user
   without passing through `agents/verifier.py`. On any error-severity
   failure, the response is `insufficient_data`, not a best-guess
   narrative.

3. **Every fact carries full provenance.** filing_url, page_number,
   table_id, row_id, is_gaap, is_restated. If a code path produces a
   fact without provenance, that path is wrong.

4. **Every inter-agent contract is a Pydantic model.** Defined in
   `backend/app/schemas.py`. No dict-shaped payloads crossing agent
   boundaries. This is the schema-constrained action generation pattern
   from Scott's pending patent applied at the workflow level.

5. **Numeric queries never use embeddings.** The Fact Retriever uses
   deterministic SQL against the canonicalized fact store. Vector search
   is scoped to prose only.

6. **Fail closed, not open.** On ambiguity, missing data, period mismatch,
   GAAP inconsistency, or sanity-band violation, the system surfaces the
   issue honestly rather than substituting a best guess.

## Do not diverge from the architectural reasoning

The write-up under `docs/` is the source of truth. Every architectural
decision (six-agent decomposition, offline vs online pipeline split, Verifier
as trust boundary, deterministic arithmetic, embeddings for prose only,
canonical fact store with ambiguity flags) is justified there against
specific failure modes of the previous system. Those decisions are
load-bearing.

**Do not modify the architecture on your own initiative.** Specifically:

- Do not add agent stages that are not in the write-up.
- Do not remove agent stages that are in the write-up. In particular, do
  not skip the Verifier because the pipeline "seems to work without it."
- Do not move arithmetic into the LLM to reduce complexity. That is the
  specific failure mode the design prevents.
- Do not add embedding-based retrieval to the numeric path to improve
  recall. That is the specific failure mode the design prevents.
- Do not convert `insufficient_data` responses to best-guess narratives
  to improve response coverage. That is the specific failure mode the
  design prevents.
- Do not skip schema validation at inter-agent boundaries because the
  JSON parse succeeded. Type-level validation is the first line of defense.
- Do not add hidden LLM calls inside deterministic stages (Fact Retriever,
  Numerical Reasoner, Verifier). Those stages exist specifically because
  they are LLM-free.
- Do not extend the metric canonicalizer to force-map ambiguous labels.
  Ambiguity flags exist specifically so uncertainty is preserved and
  surfaced to the user.

### When you want to diverge, stop and ask

If you encounter a case where the spec appears wrong (say, the schema is
missing a required field, or an agent's stated responsibility is impossible
to implement as described), do NOT silently fix it in code. Instead:

1. **Stop coding.**
2. State the specific divergence you are proposing: what the spec says,
   what you want to do instead, and why.
3. Wait for Scott to decide whether to accept, reject, or reframe.

The spec is what Scott will walk the interviewer through. If the code
diverges from what the interviewer sees in the write-up, that is a bigger
cost than a slightly imperfect design. Preserving the story matters more
than local code optimization.

## Code style

### Python
- Python 3.11+.
- `from __future__ import annotations` at the top of every module.
- Type hints on every function signature.
- Google-style docstrings on every public function.
- Structured logging with `logging.getLogger(__name__)`. No print
  statements in library code.
- Pydantic v2 syntax throughout.
- Every module has a module-level docstring stating the design principle
  it implements. See `pdf_parser.py` for the pattern.

### JavaScript and Node.js
- Node 18+.
- Vanilla JavaScript in the frontend. No build step, no bundler.
- Express only for the server. No additional frameworks.

### Style rules (universal)
- **No em dashes, en dashes, minus-sign characters, or horizontal bars
  anywhere.** Use ASCII hyphen (U+002D) only. This applies to code,
  comments, docstrings, output strings, error messages, README files,
  and any generated content. When in doubt, restructure the sentence
  with commas or parentheses.
- Sentence case in headings.
- No emoji in any deliverable.
- Prefer prose to bullets in narrative content. Prefer bullets to prose
  in reference content.

## File organization

Build modules in dependency order. Do not build downstream modules
before their dependencies are stable.

```
backend/
  app/
    schemas.py                  # 1st: Pydantic models for all inter-agent contracts
    store/
      db.py                     # 2nd: SQLite fact store with schema DDL
      vector_store.py           # 2nd: Chroma index scoped to prose only
    ingest/
      pdf_parser.py             # (reference implementation, already exists)
      table_classifier.py       # 3rd: LLM w/ schema-constrained output
      fact_extractor.py         # 3rd: LLM + rule-based validation
      canonicalizer.py          # 3rd: metric_id resolution
    agents/
      planner.py                # 4th: Query Planner (LLM)
      fact_retriever.py         # 4th: deterministic SQL
      numerical_reasoner.py     # 4th: deterministic Python
      prose_retriever.py        # 4th: vector search over prose only
      verifier.py               # 4th: gate every response
      answer_composer.py        # 4th: narrate verified facts (LLM)
    main.py                     # 5th: FastAPI wiring
    eval/
      benchmark.py              # 6th: hand-verified Q/A pairs
      runner.py                 # 6th: evaluation harness
  requirements.txt
frontend/
  server.js                     # Express proxy + static server
  package.json                  # (already exists)
  public/
    index.html
    styles.css                  # (already exists)
    app.js
tests/
  ...                           # pytest test cases
docs/
  SEC_Filing_Intelligence_TakeHome.docx
  SEC_Filing_Intelligence_Architecture.svg
```

## Testing

- pytest for backend tests.
- Every agent has unit tests using golden inputs and expected outputs.
- Every prompt is versioned in source control. Every prompt change runs
  against the golden set before merge.
- The benchmark under `backend/app/eval/` is a regression suite. It runs
  on every change to any code path in the query pipeline. New failures
  require investigation before commit.
- No test that requires network access to a live LLM provider runs in
  the default suite. Mock at the client boundary.

## Other things not to do

- Do not modify files under `docs/`. Those are Scott's; the write-up is
  the interview artifact.
- Do not add dependencies without asking. Every dependency is one more
  failure mode in production and one more thing Scott has to defend in
  the interview.
- Do not add authentication, SSO, or user management. Out of scope for
  the prototype.
- Do not add a database migration framework. SQLite plus a single DDL
  file is sufficient at prototype scope.
- Do not refactor across module boundaries without asking. Local refactors
  within a module are fine.

## When to stop and ask Scott

Beyond the divergence rule above:

- Any ambiguity about what the spec says. Read it again first. If still
  ambiguous, ask.
- Any test failure you cannot resolve within three attempts.
- Any dependency you want to add.
- Any change to the `schemas.py` contract after `schemas.py` is stable.
- Any change to the Verifier's checks after `verifier.py` is stable.
- Any performance optimization that changes the behavior of an existing test.

## Session opening protocol

At the start of every Claude Code session:

1. Read this file.
2. Read `docs/SEC_Filing_Intelligence_TakeHome.docx` if you have not
   already, or if the write-up has been updated.
3. Read the reference implementations (`pdf_parser.py`, `styles.css`,
   `package.json`) if starting work on a new module.
4. Run the existing test suite. Confirm baseline is green before making
   changes.
5. State what you intend to build in this session and wait for
   confirmation before writing code.

## Working conventions Scott holds

These are Scott's standing operational preferences, carried across every
project:

- **Ship the smallest slice that demonstrates value.** Working code beats
  architectural documents.
- **Instrument from day one, not retrofitted after prototypes.** Every
  module logs its inputs, outputs, and latency.
- **Prompts are code.** Versioned in source control, reviewed like code,
  evaluated on every change.
- **Coaching while building.** Explain the architectural reasoning behind
  your choices in commit messages and PR descriptions. Not just what
  you did, but why.

## About the discipline this file encodes

The operating discipline in this file is not ad hoc. It is the practical
application of the spec-driven development pattern Scott published in June
2026: **Tickets Don't Compile: A Practitioner's Guide to Spec-Driven
Development for Product Managers, Engineers, and AI Coding Agents**
(Amazon Publishing).

The open-source Python library that accompanies it (`specddkit` on PyPI at
`pypi.org/project/specddkit`) formalizes the spec-to-agent-ingestion
workflow that AI coding agents consume. This file is a concrete instance
of that pattern.

The principle underneath both: AI coding agents are force multipliers,
not replacements for engineering judgment. The engineer defines the spec.
The engineer reviews the output. The engineer owns the production
discipline. The agent does the mechanical work of translating spec to
code. When that division of labor holds, quality goes up and delivery
velocity goes up. When it does not hold (when the agent starts making
architectural decisions on its own initiative), quality goes down and
the interview walk-through breaks.

This file exists to keep that division of labor intact.
