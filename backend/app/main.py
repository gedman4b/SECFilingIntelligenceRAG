import os

# Must run before any downstream import touches chromadb: chromadb's
# bundled ONNX embedding function resolves Path.home() into a class
# attribute (ONNXMiniLM_L6_V2.DOWNLOAD_PATH) at class-definition time,
# i.e. the moment "import chromadb" first executes anywhere in the
# process -- not lazily per call. Vercel's deployed filesystem sets HOME
# to a read-only sandbox user directory, and HOME cannot be set as a
# Vercel project environment variable (the name is reserved), so it has
# to be patched here, as the very first lines of the actual process
# entrypoint, before app.agents.prose_retriever -> app.store.vector_store
# -> chromadb gets imported below. Confirmed against a live Vercel
# traceback: OSError: [Errno 30] Read-only file system: '/home/sbx_user...'.
if os.environ.get('VERCEL'):
    os.environ.setdefault('HOME', '/tmp')

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import QueryPlan, QueryRequest, QueryResponse, Fact
from app.agents.planner import plan_query
from app.agents.fact_retriever import (
    retrieve_facts, retrieve_ratio_facts, retrieve_ranking_facts, resolve_preferred_facts,
)
from app.agents.numerical_reasoner import compute
from app.agents.prose_retriever import retrieve_prose
from app.agents.verifier import verify
from app.agents.answer_composer import compose_answer

# Every agent and ingestion module logs at INFO via its own
# logging.getLogger(__name__), but nothing calls basicConfig() -- without a
# handler on the root logger, those calls go nowhere. main.py is the actual
# process entry point when running under uvicorn (unlike the CLI scripts,
# which each call basicConfig() in their own __main__ block), so it owns
# this instead.
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
log = logging.getLogger(__name__)

# ALLOWED_ORIGINS is a comma-separated list, e.g. "https://secfilingsint.vercel.app,http://localhost:3000".
# Defaults to local dev only so a deployed backend fails closed (blocks
# the browser, doesn't silently allow every origin) until the deployed
# frontend's actual origin is configured.
_allowed_origins_env = os.environ.get('ALLOWED_ORIGINS')
ALLOWED_ORIGINS = (
    [origin.strip() for origin in _allowed_origins_env.split(',') if origin.strip()]
    if _allowed_origins_env else ['http://localhost:3000']
)

app = FastAPI(title='SEC Filing Intelligence')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
 
@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question
    request_start = time.perf_counter()

    # Stage A: Plan
    plan: QueryPlan = plan_query(question)
 
    facts, computed, comp_expr, prose = [], None, None, []
 
    # Stage B, C: numeric path. resolved_facts collapses multiple source
    # tables for the same (metric, period) down to the preferred one(s),
    # so arithmetic and citations never silently use whichever row SQLite
    # happened to return first. verify() below still receives the raw,
    # unfiltered `facts` -- its Check 9 needs full visibility into every
    # source, agreeing or conflicting, to decide whether resolved_facts'
    # choice was actually safe to answer from.
    resolved_facts = []
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison'):
        facts = retrieve_facts(plan)
        resolved_facts = resolve_preferred_facts(facts)
        if plan.question_type in ('growth_calc', 'comparison'):
            computed, comp_expr = compute(plan, resolved_facts)

    # Stage B, C variant: margin_calc needs two metrics (numerator and
    # denominator) for the same period rather than one metric across two
    # periods. Both sides are resolved to their preferred source
    # independently, then combined: `facts` (raw, both sides) goes to
    # verify() for its conflict/period-alignment checks exactly like the
    # single-metric path, and `resolved_facts` (both sides' resolved
    # values) goes to compose_answer() so the numerator and denominator
    # are both citable.
    if plan.question_type == 'margin_calc':
        numerator_facts, denominator_facts = retrieve_ratio_facts(plan)
        facts = numerator_facts + denominator_facts
        resolved_numerator = resolve_preferred_facts(numerator_facts)
        resolved_denominator = resolve_preferred_facts(denominator_facts)
        resolved_facts = resolved_numerator + resolved_denominator
        computed, comp_expr = compute(plan, resolved_numerator, resolved_denominator)

    # Stage B, C variant: ranking has no single target metric -- it scans
    # every candidate metric (or the expense subset) for the two requested
    # periods and ranks by magnitude of change. metric_facts_by_id is
    # already resolved-to-preferred-source per metric by
    # retrieve_ranking_facts(), so both `facts` (for verify()'s checks) and
    # `resolved_facts` (for citations) are just its flattened values.
    if plan.question_type == 'ranking':
        metric_facts_by_id = retrieve_ranking_facts(plan)
        resolved_facts = [f for mfacts in metric_facts_by_id.values() for f in mfacts]
        facts = resolved_facts
        computed, comp_expr = compute(plan, facts=[], metric_facts_by_id=metric_facts_by_id)

    # Stage D: narrative path
    if plan.question_type == 'narrative':
        prose = retrieve_prose(plan, question)

    # Stage E: verify BEFORE composing
    warnings, confidence = verify(question, plan, facts, computed, prose)

    # Stage F: compose
    response = compose_answer(
        question=question,
        plan=plan,
        facts=resolved_facts,
        computed=computed,
        comp_expr=comp_expr,
        prose=prose,
        warnings=warnings,
        confidence=confidence,
    )
    log.info(
        "question=%r -> question_type=%s, confidence=%s, total latency=%.1fms",
        question, plan.question_type.value, confidence,
        (time.perf_counter() - request_start) * 1000,
    )
    return response
 
@app.get('/health')
def health():
    return {'status': 'ok'}
