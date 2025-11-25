
from pydantic import BaseModel, Field
from typing import Optional, List

class QueryEntities(BaseModel):
    """
    Defines the structured output for extracted query entities.
    """
    tickers: Optional[List[str]] = Field(None, description="Stock or ETF tickers")
    fund_names: Optional[List[str]] = Field(None, description="Full fund or company names")
    metric: Optional[str] = Field(None, description="The financial metric requested")
    timespan: Optional[str] = Field(None, description="The time period mentioned")