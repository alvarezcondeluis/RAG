import subprocess
import time
from typing import Optional, List, Dict, Any
import logging
from neomodel import db, install_all_labels, clear_neo4j_database
from .config import settings
from .models.nodes import (
    Provider, Trust, Fund, Asset, Person, Image, Risks,
    TrailingPerformance, CreditRating, Profile, AnnualPerformance,
    Sector, InvestmentStrategy, Derivative
)
from .models.relationships import (
    ManagesRel, IssuesRel, HasAssetRel, ManagedByRel, HasChartRel, HasRisksRel,
    HasPerformanceRel, HasCreditRatingRel, DefinedByRel, HasAnnualPerformanceRel,
    HasSectorAllocationRel, WithSummaryProspectusRel, HasDerivativeRel
)

logger = logging.getLogger(__name__)


class Neo4jDatabase:
    """
    Neo4j database manager for fund data using neomodel ORM with auto-start capabilities.
    
    Usage:
        db = Neo4jDatabase(auto_start=True)
        fund = db.create_fund(ticker="VTI", name="Vanguard Total Stock Market ETF")
        results = db.get_fund_by_ticker("VTI")
        db.close()
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        auto_start: bool = True
    ):
        """
        Initialize Neo4j database connection using neomodel.
        
        Args:
            uri: Neo4j connection URI (default from config)
            auto_start: Auto-start Neo4j via Docker Compose if not running
        """
        self.uri = uri or settings.NEO4J_URL
        self.docker_compose_path = settings.NEO4J_DOCKER_COMPOSE_PATH
        self.container_name = settings.NEO4J_CONTAINER_NAME
        
        if auto_start:
            self._ensure_neo4j_running()
        
        self._connect()
    
    def _is_neo4j_running(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            from neomodel import config as neomodel_config
            neomodel_config.DATABASE_URL = self.uri
            db.cypher_query("RETURN 1")
            return True
        except Exception:
            return False
    
    def _start_neo4j_docker(self) -> bool:
        """Start Neo4j via Docker Compose."""
        try:
            # Check if container is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.container_name in result.stdout:
                logger.info(f"Neo4j container {self.container_name} is already running")
                return True
            
            # Check if docker-compose.yml exists
            compose_file = settings.get_docker_compose_path()
            if not compose_file.exists():
                logger.error(f"docker-compose.yml not found at {compose_file}")
                return False
            
            # Start using docker-compose
            logger.info(f"Starting Neo4j using docker-compose from {self.docker_compose_path}")
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=str(self.docker_compose_path),
                check=True,
                capture_output=True
            )
            
            # Wait for Neo4j to be ready
            logger.info("Waiting for Neo4j to be ready...")
            for i in range(30):
                time.sleep(2)
                if self._is_neo4j_running():
                    logger.info("Neo4j is ready!")
                    return True
            
            logger.warning("Neo4j did not become ready in time")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Neo4j via docker-compose: {e}")
            return False
        except FileNotFoundError:
            logger.error("Docker or docker-compose not found. Please install Docker to use auto-start.")
            return False
    
    def _ensure_neo4j_running(self):
        """Ensure Neo4j is running, start if needed."""
        if not self._is_neo4j_running():
            logger.info("Neo4j not running, attempting to start...")
            if not self._start_neo4j_docker():
                logger.warning("Could not auto-start Neo4j. Please start it manually.")
    
    def _connect(self):
        """Establish connection to Neo4j using neomodel."""
        try:
            from neomodel import config as neomodel_config
            neomodel_config.DATABASE_URL = self.uri
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        db.close_connection()
        logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== SCHEMA SETUP ====================
    
    def install_labels(self):
        """Install all node labels and constraints using neomodel."""
        try:
            install_all_labels()
            logger.info("Installed all labels and constraints")
        except Exception as e:
            logger.warning(f"Labels may already exist: {e}")
    
    # ==================== NODE CREATION ====================
    
    def get_or_create_provider(self, name: str) -> Provider:
        """Get or create Provider node."""
        try:
            return Provider.nodes.get(name=name)
        except Provider.DoesNotExist:
            provider = Provider(name=name).save()
            logger.info(f"Created provider: {name}")
            return provider
    
    def get_or_create_trust(self, name: str) -> Trust:
        """Get or create Trust node."""
        try:
            return Trust.nodes.get(name=name)
        except Trust.DoesNotExist:
            trust = Trust(name=name).save()
            logger.info(f"Created trust: {name}")
            return trust
    
    def create_fund(self, ticker: str, name: str, **kwargs) -> Fund:
        """Create or update Fund node."""
        try:
            fund = Fund.nodes.get(ticker=ticker)
            # Update existing fund
            fund.name = name
            for key, value in kwargs.items():
                if hasattr(fund, key):
                    setattr(fund, key, value)
            fund.save()
            logger.info(f"Updated fund: {ticker}")
        except Fund.DoesNotExist:
            fund = Fund(ticker=ticker, name=name, **kwargs).save()
            logger.info(f"Created fund: {ticker}")
        return fund
    
    def get_or_create_asset(self, name: str, ticker: Optional[str] = None, **kwargs) -> Asset:
        """Get or create Asset node."""
        # Try to find by name and ticker
        assets = Asset.nodes.filter(name=name)
        if ticker:
            assets = assets.filter(ticker=ticker)
        
        try:
            asset = assets[0]
            # Update properties
            for key, value in kwargs.items():
                if hasattr(asset, key) and value:
                    setattr(asset, key, value)
            asset.save()
            return asset
        except IndexError:
            asset = Asset(name=name, ticker=ticker, **kwargs).save()
            logger.info(f"Created asset: {name}")
            return asset
    
    def get_or_create_person(self, name: str) -> Person:
        """Get or create Person node."""
        try:
            return Person.nodes.get(name=name)
        except Person.DoesNotExist:
            person = Person(name=name).save()
            logger.info(f"Created person: {name}")
            return person
    
    def get_or_create_sector(self, name: str) -> Sector:
        """Get or create Sector node."""
        try:
            return Sector.nodes.get(name=name)
        except Sector.DoesNotExist:
            sector = Sector(name=name).save()
            logger.info(f"Created sector: {name}")
            return sector
    
    def get_or_create_credit_rating(self, name: str) -> CreditRating:
        """Get or create CreditRating node."""
        try:
            return CreditRating.nodes.get(name=name)
        except CreditRating.DoesNotExist:
            rating = CreditRating(name=name).save()
            logger.info(f"Created credit rating: {name}")
            return rating
    
    # ==================== RELATIONSHIP CREATION ====================
    
    def link_provider_trust(self, provider: Provider, trust: Trust):
        """Create MANAGES relationship between Provider and Trust."""
        if not provider.manages.is_connected(trust):
            provider.manages.connect(trust)
            logger.info(f"Linked {provider.name} -> {trust.name}")
    
    def link_trust_fund(self, trust: Trust, fund: Fund):
        """Create ISSUES relationship between Trust and Fund."""
        if not trust.issues.is_connected(fund):
            trust.issues.connect(fund)
            logger.info(f"Linked {trust.name} -> {fund.ticker}")
    
    def link_fund_asset(
        self,
        fund: Fund,
        asset: Asset,
        weight: Optional[str] = None,
        n_shares: Optional[str] = None,
        date: Optional[str] = None,
        market_value: Optional[str] = None
    ):
        """Create HAS_ASSET relationship between Fund and Asset."""
        if not fund.has_assets.is_connected(asset):
            rel = fund.has_assets.connect(asset)
            if weight:
                rel.weight = weight
            if n_shares:
                rel.n_shares = n_shares
            if date:
                rel.date = date
            if market_value:
                rel.market_value = market_value
            rel.save()
    
    def link_fund_manager(
        self,
        fund: Fund,
        person: Person,
        since: Optional[str] = None,
        role: Optional[str] = None
    ):
        """Create MANAGED_BY relationship between Fund and Person."""
        if not fund.managed_by.is_connected(person):
            rel = fund.managed_by.connect(person)
            if since:
                rel.since = since
            if role:
                rel.role = role
            rel.save()
    
    def link_fund_sector(
        self,
        fund: Fund,
        sector: Sector,
        weight: Optional[str] = None
    ):
        """Create HAS_SECTOR_ALLOCATION relationship."""
        if not fund.has_sector_allocation.is_connected(sector):
            rel = fund.has_sector_allocation.connect(sector)
            if weight:
                rel.weight = weight
                rel.save()
    
    def link_fund_credit_rating(
        self,
        fund: Fund,
        rating: CreditRating,
        weight: Optional[str] = None
    ):
        """Create HAS_CREDIT_RATING relationship."""
        if not fund.has_credit_ratings.is_connected(rating):
            rel = fund.has_credit_ratings.connect(rating)
            if weight:
                rel.weight = weight
                rel.save()
    
    # ==================== COMPLEX NODE CREATION ====================
    
    def create_risks_node(
        self,
        fund: Fund,
        content: str,
        date: Optional[str] = None
    ) -> Risks:
        """Create Risks node and link to Fund."""
        risks = Risks(content=content).save()
        rel = fund.has_risks.connect(risks)
        if date:
            rel.date = date
            rel.save()
        return risks
    
    def create_performance_node(
        self,
        fund: Fund,
        return_1y: Optional[str] = None,
        return_5y: Optional[str] = None,
        return_10y: Optional[str] = None,
        return_inception: Optional[str] = None,
        date: Optional[str] = None
    ) -> TrailingPerformance:
        """Create TrailingPerformance node and link to Fund."""
        performance = TrailingPerformance(
            return_1y=return_1y,
            return_5y=return_5y,
            return_10y=return_10y,
            return_inception=return_inception
        ).save()
        rel = fund.has_performance.connect(performance)
        if date:
            rel.date = date
            rel.save()
        return performance
    
    def create_annual_performance_node(
        self,
        fund: Fund,
        year: str,
        **kwargs
    ) -> AnnualPerformance:
        """Create AnnualPerformance node and link to Fund."""
        annual_perf = AnnualPerformance(year=year, **kwargs).save()
        fund.has_annual_performance.connect(annual_perf)
        return annual_perf
    
    def create_strategy_node(
        self,
        fund: Fund,
        full_text_md: str,
        objective: Optional[str] = None,
        strategies: Optional[str] = None,
        date: Optional[str] = None
    ) -> InvestmentStrategy:
        """Create InvestmentStrategy node and link to Fund."""
        strategy = InvestmentStrategy(
            full_text_md=full_text_md,
            objective=objective,
            strategies=strategies
        ).save()
        rel = fund.with_summary_prospectus.connect(strategy)
        if date:
            rel.date = date
            rel.save()
        return strategy
    
    def create_profile_node(
        self,
        fund: Fund,
        text: str,
        embedding: Optional[List[float]] = None
    ) -> Profile:
        """Create Profile node with embeddings and link to Fund."""
        profile = Profile(text=text, embedding=embedding).save()
        fund.defined_by.connect(profile)
        return profile
    
    def create_derivative_node(
        self,
        fund: Fund,
        name: str,
        fair_value_lvl: Optional[str] = None,
        derivative_type: Optional[str] = None,
        weight: Optional[str] = None,
        date: Optional[str] = None,
        termination_date: Optional[str] = None,
        unrealized_pnl: Optional[str] = None,
        amount: Optional[str] = None
    ) -> Derivative:
        """Create Derivative node and link to Fund."""
        # Try to find existing derivative
        try:
            derivative = Derivative.nodes.get(name=name)
        except Derivative.DoesNotExist:
            derivative = Derivative(
                name=name,
                fair_value_lvl=fair_value_lvl,
                type=derivative_type
            ).save()
        
        if not fund.has_derivatives.is_connected(derivative):
            rel = fund.has_derivatives.connect(derivative)
            if weight:
                rel.weight = weight
            if date:
                rel.date = date
            if termination_date:
                rel.termination_date = termination_date
            if unrealized_pnl:
                rel.unrealized_pnl = unrealized_pnl
            if amount:
                rel.amount = amount
            rel.save()
        
        return derivative
    
    # ==================== QUERY METHODS ====================
    
    def get_fund_by_ticker(self, ticker: str) -> Optional[Fund]:
        """Get fund by ticker."""
        try:
            return Fund.nodes.get(ticker=ticker)
        except Fund.DoesNotExist:
            return None
    
    def get_fund_with_holdings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get fund with all its holdings."""
        try:
            fund = Fund.nodes.get(ticker=ticker)
            holdings = []
            
            for asset in fund.has_assets.all():
                rel = fund.has_assets.relationship(asset)
                holdings.append({
                    "asset": asset,
                    "weight": rel.weight,
                    "n_shares": rel.n_shares,
                    "market_value": rel.market_value,
                    "date": rel.date,
                })
            
            return {
                "fund": fund,
                "holdings": holdings
            }
        except Fund.DoesNotExist:
            return None
    
    def get_all_funds(self) -> List[Fund]:
        """Get all funds."""
        return list(Fund.nodes.order_by('ticker'))
    
    def get_funds_by_provider(self, provider_name: str) -> List[Fund]:
        """Get all funds managed by a provider."""
        try:
            provider = Provider.nodes.get(name=provider_name)
            funds = []
            for trust in provider.manages.all():
                funds.extend(trust.issues.all())
            return funds
        except Provider.DoesNotExist:
            return []
    
    def search_funds_by_name(self, name_pattern: str) -> List[Fund]:
        """Search funds by name pattern."""
        return list(Fund.nodes.filter(name__icontains=name_pattern).order_by('ticker'))
    
    def delete_fund(self, ticker: str):
        """Delete fund and all its relationships."""
        try:
            fund = Fund.nodes.get(ticker=ticker)
            fund.delete()
            logger.info(f"Deleted fund: {ticker}")
        except Fund.DoesNotExist:
            logger.warning(f"Fund {ticker} not found")
    
    def clear_database(self):
        """Clear entire database (use with caution!)."""
        clear_neo4j_database(db)
        logger.warning("Database cleared!")
