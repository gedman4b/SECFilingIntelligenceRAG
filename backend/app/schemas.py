"""
Every inter-agent contract is expressed as a Pydantic model. This is the schema-constrained action generation 
pattern from my pending patent applied at the workflow level: no agent output enters the next stage without 
validation against a registered schema.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum
 
class QuestionType(str, Enum):
    NUMERIC_LOOKUP = 'numeric_lookup'
    GROWTH_CALC = 'growth_calc'
    NARRATIVE = 'narrative'
    COMPARISON = 'comparison'
    # A ratio between two different metrics in the same period, e.g.
    # "gross margin" (gross profit / revenue) or "SG&A as a percentage of
    # revenue". Distinct from growth_calc/comparison, which both relate
    # ONE metric across two periods; this relates TWO metrics in the SAME
    # period(s).
    MARGIN_CALC = 'margin_calc'
    # Ranks multiple canonical metrics by magnitude of change between two
    # periods, e.g. "which metrics deteriorated most" or "biggest expense
    # increases". Distinct from growth_calc, which computes ONE named
    # metric's change; this has no single target metric, only a company
    # and two periods.
    RANKING = 'ranking'
    UNKNOWN = 'unknown'
 
class Period(BaseModel):
    year: int
    quarter: Optional[int] = None  # None = full year
    is_ttm: bool = False
    # True when this period is a year-to-date cumulative figure through the
    # stated quarter (e.g. "six months ended"), not the discrete quarter
    # itself and not the full year. A 10-Q's condensed statements routinely
    # report both a discrete quarter and a YTD column side by side; without
    # this flag both would collide under quarter=None with the true
    # full-year figure. quarter is never None when is_ytd is True.
    is_ytd: bool = False
 
class QueryRequest(BaseModel):
    question: str

class QueryPlan(BaseModel):
    question_type: QuestionType
    company_ticker: Optional[str] = None
    metric_natural_language: Optional[str] = None
    metric_canonical_id: Optional[str] = None
    # Only set when question_type is margin_calc. metric_natural_language /
    # metric_canonical_id above is the ratio's numerator (e.g. "gross
    # profit"); these are the denominator (e.g. "revenue"). Left for the
    # deterministic canonicalizer to resolve, same as the numerator.
    ratio_denominator_natural_language: Optional[str] = None
    ratio_denominator_canonical_id: Optional[str] = None
    # Only set when question_type is ranking. top_increase/top_decrease
    # rank by raw growth % regardless of what the metric represents;
    # most_deteriorated/most_improved instead account for each metric's
    # is_expense flag (ingest/canonicalizer.py), so a rising expense and a
    # falling revenue both count as deterioration rather than whichever has
    # the larger raw percentage.
    ranking_direction: Literal['top_increase', 'top_decrease', 'most_deteriorated', 'most_improved'] = 'top_increase'
    # Only set when question_type is ranking. 'expense' restricts ranked
    # candidates to cost/expense-line metrics (e.g. "biggest expense
    # increases"); None ranks every canonical metric with data in both
    # requested periods.
    ranking_scope: Optional[Literal['expense']] = None
    periods: List[Period] = Field(default_factory=list)
    gaap_preference: Literal['gaap', 'non_gaap', 'either'] = 'gaap'
    ambiguity_flags: List[str] = Field(default_factory=list)
 
class Fact(BaseModel):
    value: float
    units: str  # 'USD', 'USD_millions', 'percent', 'shares'
    period: Period
    metric_canonical_id: str
    metric_raw_label: str
    is_gaap: bool
    is_restated: bool = False
    # False when the source filing's financial statements are unaudited,
    # per standard SEC convention: a 10-K's annual statements are audited,
    # a 10-Q's quarterly statements are not. Derived deterministically from
    # the filing's form_type, not guessed by an LLM. Named explicitly in
    # the assignment brief's ambiguity checklist.
    is_audited: bool = True
    filing_id: str
    filing_url: str
    page_number: int
    table_id: str
    row_id: int
    ambiguity_flags: List[str] = Field(default_factory=list)
    # True when this fact's source table resolves more distinct canonical
    # metrics than any other table contributing to the same (metric,
    # period): empirically, the true primary financial statement resolves
    # many metrics from one table, while MD&A commentary and footnotes that
    # happen to restate the same figure resolve only one or two. Used to
    # pick a single citation among multiple agreeing sources without
    # suppressing them from the Verifier's conflict check (Check 9), which
    # still needs to see every source, agreeing or not.
    is_preferred_source: bool = False
 
class Warning(BaseModel):
    severity: Literal['info', 'warning', 'error']
    message: str
    field: Optional[str] = None
 
class QueryResponse(BaseModel):
    answer_text: str
    raw_facts: List[Fact]
    computed_value: Optional[float] = None
    computation_expression: Optional[str] = None
    warnings: List[Warning] = Field(default_factory=list)
    citations: List[dict]  # {filing_url, page, table_id, label, value}
    confidence: Literal['high', 'medium', 'low', 'insufficient_data']
