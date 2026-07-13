# SEC Filing Intelligence backend

FastAPI backend for the SEC Filing Intelligence prototype: a six-agent pipeline that answers natural-language financial questions over a small corpus of Tesla and Apple SEC filings, with full provenance and honest uncertainty.

The architectural reasoning lives in `docs/SEC_Filing_Intelligence_TakeHome.docx`. Operating rules for anyone (human or agent) working in this codebase live in `AGENTS.md` at the repository root. This file is practical setup and run instructions only.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Create a `.env` file in `backend/` with an Anthropic API key. The offline ingestion pipeline (table classification, fact extraction) and the online Query Planner and Answer Composer all call the Anthropic API.

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Running the server

```
./run_server.sh
```

This loads `.env` and starts uvicorn with autoreload on `http://localhost:8000`. Equivalent to:

```
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`GET /health` returns `{"status": "ok"}`. `POST /query` takes a JSON body `{"question": "..."}` and returns a `QueryResponse` (answer text, raw facts, warnings, citations, confidence).

The frontend (`../frontend/`) is a separate Node/Express app that proxies `/api/*` to this server. Run it with `npm start` from `frontend/` on port 3000.

## Running the test suite

```
uv run pytest
```

108 unit tests across every ingestion and agent module, using golden inputs and expected outputs. The Anthropic client is mocked at the boundary in every test that touches an LLM-calling module (`planner`, `answer_composer`, `table_classifier`, `fact_extractor`), so the suite makes no live API calls and runs in a few seconds. Tests that touch the vector store use chromadb's bundled local embedding model, which is not a live LLM provider.

Every test uses a temporary SQLite/Chroma store (`tmp_path`), never the real persistent fact store.

## Populating the fact store

The persistent fact store (`app/store/fact_store.db`, `app/store/chroma_data/`) is not checked in. To populate it, run the offline ingestion pipeline over the six filings in `app/ingest/pdfs/`:

```
python -m app.ingest.run_ingestion
```

This makes real Anthropic API calls (table classification and fact extraction, once per table) and is slow: expect several minutes and real API cost for the full run. To re-ingest a single filing instead of all six:

```
python -m app.ingest.run_ingestion --only TSLA-10K-2025-12-31
```

Filing IDs are listed in `app/ingest/run_ingestion.py`'s `FILINGS` list. Use `--db-path` / `--chroma-dir` to point at a different store instead of the default persistent one.

## Running the evaluation benchmark

```
python -m app.eval.runner
```

Runs the 20 hand-verified question/answer pairs in `app/eval/benchmark.py` against the deterministic query pipeline (Fact Retriever, Numerical Reasoner, Verifier) using each case's golden plan directly, never calling the Query Planner or Answer Composer. Requires a populated fact store; run ingestion first. Narrative cases are marked for human relevance review rather than auto-scored pass/fail, per the write-up's own definition of narrative correctness.

Exits non-zero if any non-narrative case fails, so it is safe to use as a CI gate on changes to the query pipeline.

## Project structure

```
app/
  schemas.py            Pydantic models for every inter-agent contract
  main.py                FastAPI app: POST /query, GET /health
  instrumentation.py     Shared latency-logging decorator
  store/
    db.py                 SQLite fact store
    vector_store.py        Chroma index over prose only
  ingest/                 Offline pipeline: parse, classify, extract, canonicalize
    run_ingestion.py       Orchestrates the above over app/ingest/pdfs/
  agents/                 Online pipeline: plan, retrieve, compute, verify, compose
  eval/                   Benchmark cases and evaluation harness
tests/                    pytest suite, mirrors app/ one file per module
```
