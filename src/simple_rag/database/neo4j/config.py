import os
from dotenv import load_dotenv
from neomodel import config as neomodel_config
from pathlib import Path

# Load variables from .env file
load_dotenv()

class Config:
    # Neo4j Settings (matching docker-compose.yml credentials)
    NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:luis2014@localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "luis2014")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Docker compose path
    NEO4J_DOCKER_COMPOSE_PATH = Path(__file__).parent.parent.parent.parent.parent / "neo4j"
    NEO4J_CONTAINER_NAME = "fund_graph_db"
    
    # RAG Settings (e.g., embedding model dimensions)
    EMBEDDING_DIMENSION = 1536
    
    @classmethod
    def setup_neomodel(cls):
        """Configure neomodel with connection settings."""
        neomodel_config.DATABASE_URL = cls.NEO4J_URL
        neomodel_config.DATABASE_NAME = cls.NEO4J_DATABASE
    
    @classmethod
    def get_docker_compose_path(cls) -> Path:
        """Get the path to the docker-compose.yml file."""
        return cls.NEO4J_DOCKER_COMPOSE_PATH / "docker-compose.yml"

settings = Config()