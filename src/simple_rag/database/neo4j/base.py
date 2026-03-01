import subprocess
import time
import os
from typing import Optional, List, Dict, Any
from datetime import date
import logging
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from .config import settings

logger = logging.getLogger(__name__)


class Neo4jDatabaseBase:
    """
    Base class for Neo4j database operations.
    Handles connection management, Docker startup, and query execution.
    
    Usage:
        db = Neo4jDatabaseBase(auto_start=True)
        results = db._execute_query("MATCH (n) RETURN n LIMIT 10")
        db.close()
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auto_start: bool = True
    ):
        """
        Initialize Neo4j database connection using native driver.
        
        Args:
            uri: Neo4j connection URI (default from config)
            username: Neo4j username (default from config)
            password: Neo4j password (default from config)
            auto_start: Auto-start Neo4j via Docker Compose if not running
        """
        self.uri = uri or settings.NEO4J_URL
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.container_name = settings.NEO4J_CONTAINER_NAME
        self.driver: Optional[Driver] = None

        if auto_start:
            self._ensure_neo4j_running()
        
        # Ensure Neo4j is actually ready before connecting
        max_wait_attempts = 10
        for attempt in range(max_wait_attempts):
            if self._is_neo4j_running():
                break
            if attempt < max_wait_attempts - 1:  # Don't sleep on last attempt
                logger.info(f"Waiting for Neo4j to be ready... (attempt {attempt + 1}/{max_wait_attempts})")
                time.sleep(3)
        
        self._connect()
    
    def _is_neo4j_running(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            test_driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            test_driver.verify_connectivity()
            test_driver.close()
            return True
        except (ServiceUnavailable, AuthError, Exception):
            return False
    
    def _start_neo4j_docker(self) -> bool:
        """Start Neo4j via Docker Compose."""
        try:
            
            # Check if container is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            # Check for permission errors
            if result.returncode != 0:
                error_msg = result.stderr.lower() if result.stderr else ""
                if "permission denied" in error_msg or "docker.sock" in error_msg:
                    logger.error("Docker permission denied. Your user is not in the 'docker' group.")
                    logger.error("To fix this, run the following commands:")
                    logger.error("  1. sudo usermod -aG docker $USER")
                    logger.error("  2. Log out and log back in (or reboot)")
                    logger.error("  3. Verify with: groups | grep docker")
                    logger.error("\nAlternatively, start Neo4j manually with:")
                    logger.error(f"  cd {settings.NEO4J_DOCKER_COMPOSE_PATH} && docker compose up -d")
                    return False
                else:
                    logger.error(f"Docker command failed: {result.stderr}")
                    return False
            
            if self.container_name in result.stdout:
                logger.info(f"Neo4j container {self.container_name} is already running")
                return True
            
            # Check if docker-compose.yml exists
            compose_file = settings.get_docker_compose_path()
            if not compose_file.exists():
                logger.error(f"docker-compose.yml not found at {compose_file}")
                return False
            project_dir = os.path.dirname(compose_file)

            # Start using docker-compose
            logger.info(f"Starting Neo4j using docker-compose from {compose_file}")
            result = subprocess.run(
                [
                    "docker", "compose", 
                    "-f", str(compose_file),       # Point to the specific file
                    "--project-directory", project_dir, # Ensure volumes map correctly
                    "up", "-d"                     # Run in background
                ], 
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.lower() if result.stderr else ""
                if "permission denied" in error_msg or "docker.sock" in error_msg:
                    logger.error("Docker permission denied. Your user is not in the 'docker' group.")
                    logger.error("To fix this, run: sudo usermod -aG docker $USER")
                    logger.error("Then log out and log back in.")
                else:
                    logger.error(f"Failed to start Neo4j: {result.stderr}")
                return False
            
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
        else:
            logger.info("Neo4j is already running")
    
    def _connect(self):
        """Establish connection to Neo4j using native driver."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600
            )
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}. Is Neo4j running?")
            logger.error(f"Try: docker ps to check if Neo4j container is running")
            raise
        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j. Check username/password.")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.error(f"Connection URI: {self.uri}")
            logger.error(f"Username: {self.username}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def _execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write Cypher query in a transaction."""
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: list(tx.run(query, parameters or {})))
            return [record.data() for record in result]
