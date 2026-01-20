@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations"""
    # Vector search params
    vector_top_k: int = 10
    vector_min_score: float = 0.75
    
    # Keyword search params
    keyword_top_k: int = 10
    keyword_min_score: float = 0.5
    
    # Hybrid scoring weights
    keyword_weight: float = 0.25
    vector_weight: float = 0.40
    graph_weight: float = 0.35
    
    # Filters
    max_expense_ratio: Optional[float] = None
    min_net_assets: Optional[float] = None
    min_inception_years: Optional[int] = None
    
    # Reranking
    use_rrf: bool = False  # Reciprocal Rank Fusion
    rrf_k: int = 60
    
    # Results
    final_top_k: int = 10
    include_metadata: bool = True
    include_context: bool = True  # Include related entities
