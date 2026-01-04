from neomodel import StructuredRel, StringProperty


class ManagesRel(StructuredRel):
    """Provider MANAGES Trust"""
    pass


class IssuesRel(StructuredRel):
    """Trust ISSUES Fund"""
    pass


class HasAssetRel(StructuredRel):
    """Fund HAS_ASSET Asset"""
    weight = StringProperty()
    n_shares = StringProperty()
    date = StringProperty()
    market_value = StringProperty()


class ManagedByRel(StructuredRel):
    """Fund MANAGED_BY Person"""
    since = StringProperty()
    role = StringProperty()


class HasChartRel(StructuredRel):
    """Fund HAS_CHART Image"""
    pass


class HasRisksRel(StructuredRel):
    """Fund HAS_RISKS Risks"""
    date = StringProperty()


class HasPerformanceRel(StructuredRel):
    """Fund HAS_PERFORMANCE TrailingPerformance"""
    date = StringProperty()


class HasCreditRatingRel(StructuredRel):
    """Fund HAS_CREDIT_RATING CreditRating"""
    weight = StringProperty()


class DefinedByRel(StructuredRel):
    """Fund DEFINED_BY Profile"""
    pass


class HasAnnualPerformanceRel(StructuredRel):
    """Fund HAS_ANNUAL_PERFORMANCE AnnualPerformance"""
    pass


class HasSectorAllocationRel(StructuredRel):
    """Fund HAS_SECTOR_ALLOCATION Sector"""
    weight = StringProperty()


class WithSummaryProspectusRel(StructuredRel):
    """Fund WITH_SUMMARY_PROSPECTUS InvestmentStrategy"""
    date = StringProperty()


class HasDerivativeRel(StructuredRel):
    """Fund HAS_DERIVATIVE Derivative"""
    weight = StringProperty()
    date = StringProperty()
    termination_date = StringProperty()
    unrealized_pnl = StringProperty()
    amount = StringProperty()
