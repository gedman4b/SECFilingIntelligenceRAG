""" The Planner uses an LLM with a strict Pydantic schema to convert natural language into a structured plan. 
The schema is enforced by function-calling / structured-output modes, so the LLM cannot return free text 
where a typed field is expected.
"""
import anthropic
from app.schemas import QueryPlan, QuestionType, Period
import json
 
client = anthropic.Anthropic()
 
SYSTEM_PROMPT = '''You classify SEC filing questions into structured plans.
You must return a JSON object matching the QueryPlan schema exactly.
Rules:
1. If the question asks for a growth rate, delta, or year-over-year change, use question_type=growth_calc and include BOTH periods.
2. If the question asks for a single value at a point in time, use question_type=numeric_lookup.
3. If the question asks about management commentary, risks, or narrative content, use question_type=narrative.
4. Never invent a company or period. If unclear, add an ambiguity_flag.
5. Default gaap_preference to "gaap" unless the question says "non-GAAP" or "adjusted".
'''
 
def plan_query(question: str) -> QueryPlan:
    response = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': question}],
    )
    raw = response.content[0].text
    # Extract JSON block; enforce schema via Pydantic
    try:
        data = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
        return QueryPlan(**data)
    except Exception as e:
        # Fail closed: unknown plan instead of guessing
        return QueryPlan(
            question_type=QuestionType.UNKNOWN,
            ambiguity_flags=[f'plan_parse_failed: {e}'],
        )
