from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import QueryPlan, QueryResponse, Fact
from app.agents.planner import plan_query
from app.agents.fact_retriever import retrieve_facts
from app.agents.numerical_reasoner import compute
from app.agents.prose_retriever import retrieve_prose
from app.agents.verifier import verify
from app.agents.answer_composer import compose_answer
 
app = FastAPI(title='SEC Filing Intelligence')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
 
@app.post('/query', response_model=QueryResponse)
def query(question: str) -> QueryResponse:
    # Stage A: Plan
    plan: QueryPlan = plan_query(question)
 
    facts, computed, comp_expr, prose = [], None, None, []
 
    # Stage B, C: numeric path
    if plan.question_type in ('numeric_lookup', 'growth_calc', 'comparison'):
        facts = retrieve_facts(plan)
        if plan.question_type in ('growth_calc', 'comparison'):
            computed, comp_expr = compute(plan, facts)
 
    # Stage D: narrative path
    if plan.question_type == 'narrative':
        prose = retrieve_prose(plan, question)
 
    # Stage E: verify BEFORE composing
    warnings, confidence = verify(question, plan, facts, computed, prose)
 
    # Stage F: compose
    response = compose_answer(
        question=question,
        plan=plan,
        facts=facts,
        computed=computed,
        comp_expr=comp_expr,
        prose=prose,
        warnings=warnings,
        confidence=confidence,
    )
    return response
 
@app.get('/health')
def health():
    return {'status': 'ok'}
