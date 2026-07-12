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
    UNKNOWN = 'unknown'
 
class Period(BaseModel):
    year: int
    quarter: Optional[int] = None  # None = full year
    is_ttm: bool = False
 
class QueryPlan(BaseModel):
    question_type: QuestionType
    company_ticker: Optional[str] = None
    metric_natural_language: Optional[str] = None
    metric_canonical_id: Optional[str] = None
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
    filing_id: str
    filing_url: str
    page_number: int
    table_id: str
    row_id: int
    ambiguity_flags: List[str] = Field(default_factory=list)
 
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
