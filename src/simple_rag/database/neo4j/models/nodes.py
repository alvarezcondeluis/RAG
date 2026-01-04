from neomodel import (
    StructuredNode, StringProperty, ArrayProperty, FloatProperty,
    RelationshipTo, RelationshipFrom, UniqueIdProperty, DateProperty, IntegerProperty, RegexProperty
)
from typing import Optional, List


class Provider(StructuredNode):
    """Investment provider/company (e.g., Vanguard, BlackRock)"""
    name = StringProperty(unique_index=True, required=True)
    
    # Relationships
    manages = RelationshipTo('Trust', 'MANAGES')


class Trust(StructuredNode):
    """Investment trust entity"""
    name = StringProperty(unique_index=True, required=True)
    
    # Relationships
    managed_by = RelationshipFrom('Provider', 'MANAGES')
    issues = RelationshipTo('Fund', 'ISSUES')


class ShareClass(StructuredNode):
    """
    Represents a type of share (e.g., 'Admiral Shares').
    Connecting funds to this node helps the LLM understand the fee structure logic.
    """
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    description = StringProperty()
    # Reverse relationship (Find all funds that have this class)
    funds = RelationshipFrom('Fund', 'HAS_SHARE_CLASS')

class Fund(StructuredNode):
    """Main fund entity with all key attributes"""
    ticker = RegexPropertyProperty(expression=r'^[A-Z]{1,6}$', unique_index=True, required=True)
    name = StringProperty(required=True, index=True)
    expense_ratio = FloatProperty()
    inception_date = DateProperty()
    share_class = RelationshipTo('ShareClass', 'HAS_SHARE_CLASS')
    net_assets = FloatProperty()
    turnover_rate = FloatProperty()
    advisory_fee = FloatProperty()
    holdings = IntegerProperty()
    costs_10k = FloatProperty()
    benchmark = StringProperty()
    # Relationships
    issued_by = RelationshipFrom('Trust', 'ISSUES')
    has_assets = RelationshipTo('Asset', 'HAS_ASSET')
    managed_by = RelationshipTo('Person', 'MANAGED_BY')
    has_charts = RelationshipTo('Image', 'HAS_CHART')
    has_risks = RelationshipTo('Risks', 'HAS_RISKS')
    has_performance = RelationshipTo('TrailingPerformance', 'HAS_PERFORMANCE')
    has_credit_ratings = RelationshipTo('CreditRating', 'HAS_CREDIT_RATING')
    defined_by = RelationshipTo('Profile', 'DEFINED_BY')
    has_annual_performance = RelationshipTo('AnnualPerformance', 'HAS_ANNUAL_PERFORMANCE')
    has_sector_allocation = RelationshipTo('Sector', 'HAS_SECTOR_ALLOCATION')
    with_summary_prospectus = RelationshipTo('InvestmentStrategy', 'WITH_SUMMARY_PROSPECTUS')
    has_derivatives = RelationshipTo('Derivative', 'HAS_DERIVATIVE')


class Asset(StructuredNode):
    """Portfolio holding/asset (stocks, bonds, etc.)"""
    name = StringProperty(required=True)
    ticker = StringProperty(index=True)
    category = StringProperty(index=True)
    issuer_category = StringProperty()
    country = StringProperty()
    sector = StringProperty()
    currency = StringProperty()
    
    # Relationships
    held_by_funds = RelationshipFrom('Fund', 'HAS_ASSET')


class Person(StructuredNode):
    """Fund manager or portfolio manager"""
    name = StringProperty(unique_index=True, required=True)
    
    # Relationships
    manages_funds = RelationshipFrom('Fund', 'MANAGED_BY')


class Image(StructuredNode):
    """Chart or visualization image"""
    category = StringProperty(required=True)
    date = StringProperty()
    url = StringProperty()
    
    # Relationships
    fund = RelationshipFrom('Fund', 'HAS_CHART')


class Risks(StructuredNode):
    """Fund risk information"""
    content = StringProperty(required=True)
    
    # Relationships
    fund = RelationshipFrom('Fund', 'HAS_RISKS')


class TrailingPerformance(StructuredNode):
    """Trailing returns performance data"""
    return_1y = StringProperty()
    return_5y = StringProperty()
    return_10y = StringProperty()
    return_inception = StringProperty()
    
    # Relationships
    fund = RelationshipFrom('Fund', 'HAS_PERFORMANCE')


class CreditRating(StructuredNode):
    """Credit rating category (e.g., AAA, AA, BBB)"""
    name = StringProperty(unique_index=True, required=True)
    
    # Relationships
    funds = RelationshipFrom('Fund', 'HAS_CREDIT_RATING')


class Profile(StructuredNode):
    """Fund profile with embeddings for RAG"""
    text = StringProperty(required=True)
    embedding = ArrayProperty(FloatProperty())
    
    # Relationships
    fund = RelationshipFrom('Fund', 'DEFINED_BY')


class AnnualPerformance(StructuredNode):
    """Annual performance and financial highlights"""
    year = StringProperty(required=True)
    return_pct = StringProperty()
    expense_ratio = StringProperty()
    turnover_rate = StringProperty()
    net_assets = StringProperty()
    distribution_shares = StringProperty()
    nav_beginning = StringProperty()
    nav_ending = StringProperty()
    
    # Relationships
    fund = RelationshipFrom('Fund', 'HAS_ANNUAL_PERFORMANCE')


class Sector(StructuredNode):
    """Sector category (e.g., Technology, Healthcare)"""
    name = StringProperty(unique_index=True, required=True)
    
    # Relationships
    funds = RelationshipFrom('Fund', 'HAS_SECTOR_ALLOCATION')


class InvestmentStrategy(StructuredNode):
    """Investment strategy and summary prospectus content"""
    full_text_md = StringProperty(required=True)
    objective = StringProperty()
    strategies = StringProperty()
    
    # Relationships
    fund = RelationshipFrom('Fund', 'WITH_SUMMARY_PROSPECTUS')


class Derivative(StructuredNode):
    """Derivative instrument (options, futures, swaps)"""
    name = StringProperty(required=True)
    fair_value_lvl = StringProperty()
    type = StringProperty()
    
    # Relationships
    funds = RelationshipFrom('Fund', 'HAS_DERIVATIVE')
