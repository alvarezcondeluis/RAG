from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct
from tqdm import tqdm
import subprocess
import time
import requests
import os

def batch_iterate(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class QdrantDatabase:
    def __init__(self, collection_name, vector_size=1024, batch_size=512, auto_start_qdrant=True):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.qdrant_process = None
        
        # Check and start Qdrant if needed
        if auto_start_qdrant:
            self._ensure_qdrant_running()
        
        self.qdrant_client = QdrantClient(url="http://localhost:6334", prefer_grpc=True)

    def _is_qdrant_running(self):
        """Check if Qdrant server is running."""
        try:
            response = requests.get("http://localhost:6333/", timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _start_qdrant_server(self):
        """Start Qdrant server in a subprocess using Docker."""
        try:
            # Check if Docker is available
            docker_check = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            
            if docker_check.returncode != 0:
                print("✗ Docker not found. Please install Docker to auto-start Qdrant.")
                print("  Or start Qdrant manually: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
                return False
            
            print("Starting Qdrant server with Docker...")
            
            # Check if container already exists
            check_container = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=qdrant", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "qdrant" in check_container.stdout:
                # Container exists, start it
                print("  - Found existing Qdrant container, starting it...")
                subprocess.run(["docker", "start", "qdrant"], check=True, timeout=10)
            else:
                # Create and start new container
                print("  - Creating new Qdrant container...")
                self.qdrant_process = subprocess.Popen(
                    [
                        "docker", "run", "--name", "qdrant",
                        "-p", "6333:6333",
                        "-p", "6334:6334",
                        "-d",
                        "qdrant/qdrant"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.qdrant_process.wait(timeout=30)
            
            # Wait for server to be ready (max 30 seconds)
            max_wait = 30
            wait_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                if self._is_qdrant_running():
                    print(f"✓ Qdrant server started successfully (took {elapsed}s)")
                    return True
                time.sleep(wait_interval)
                elapsed += wait_interval
            
            print("⚠ Qdrant server started but not responding yet")
            return False
            
        except subprocess.TimeoutExpired:
            print("✗ Timeout while starting Qdrant server")
            return False
        except FileNotFoundError:
            print("✗ Docker command not found. Please install Docker first.")
            print("  Visit: https://docs.docker.com/get-docker/")
            return False
        except Exception as e:
            print(f"✗ Failed to start Qdrant server: {e}")
            return False
    
    def _ensure_qdrant_running(self):
        """Ensure Qdrant server is running, start it if not."""
        if self._is_qdrant_running():
            print("✓ Qdrant server is already running")
            return True
        
        print("Qdrant server not detected, attempting to start...")
        return self._start_qdrant_server()
    
    def stop_qdrant_server(self):
        """Stop the Qdrant Docker container if it was started by this instance."""
        try:
            print("Stopping Qdrant server...")
            subprocess.run(["docker", "stop", "qdrant"], timeout=10, check=True)
            print("✓ Qdrant server stopped")
        except Exception as e:
            print(f"⚠ Could not stop Qdrant server: {e}")

    def create_collection(self, name=None):
        name = name or self.collection_name
        if not self.qdrant_client.collection_exists(name):
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                )
            )

    def batch_upsert(self, embeddings_list, name=None):
        name = name or self.collection_name
        """
        embeddings_list: List of dictionaries with structure:
        {
            'vector': [...],
            'payload': {
                'text': '...',
                'source_document': '...',
                'page_number': ...,
                'chunk_type': '...',
                'section_title': '...'
            }
        }
        """
        point_id = 0
        
        for batch in tqdm(
            batch_iterate(embeddings_list, self.batch_size),
            total=len(embeddings_list) // self.batch_size + (1 if len(embeddings_list) % self.batch_size else 0),
            desc="Ingesting in batches"
        ):
            points = []
            for embedding_entry in batch:
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding_entry['vector'],  # Extract the actual vector
                        payload=embedding_entry['payload']  # Use the payload as-is
                    )
                )
                point_id += 1
            
            # Use upsert instead of upload_collection
            self.qdrant_client.upsert(
                collection_name=name,
                points=points
            )
        
        # Update indexing threshold after all data is uploaded
        self.qdrant_client.update_collection(
            collection_name=name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

    def create_payload_indexes(self, name=None):
        """Creates indexes on the metadata fields for fast filtering."""
        name = name or self.collection_name
        try:
            # Index for exact string matching on filename, path, and type
            self.qdrant_client.create_payload_index(
                collection_name=name,
                field_name="metadata.name", # Use dot notation for nested fields
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.qdrant_client.create_payload_index(
                collection_name=name,
                field_name="metadata.type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.qdrant_client.create_payload_index(
                collection_name=name,
                field_name="metadata.risk",
                field_schema=models.PayloadSchemaType.KEYWORD.INTEGER
            )

            print("✅ Successfully created payload indexes.")
        except Exception as e:
            print(f"❌ Error creating payload indexes: {e}")

    def delete_all_data(self, name=None):
        """
        Deletes all data (points) from the collection while keeping the collection structure intact.
        This is useful for clearing the database without recreating the collection.
        """
        name = name or self.collection_name
        try:
            # Check if collection exists first
            if not self.qdrant_client.collection_exists(name):
                print(f"Collection '{name}' does not exist.")
                return False
            
            # Get collection info to check if it has any points
            collection_info = self.qdrant_client.get_collection(name)
            points_count = collection_info.points_count
            
            if points_count == 0:
                print(f"Collection '{name}' is already empty.")
                return True
            
            print(f"Deleting {points_count} points from collection '{name}'...")
            
            # Delete all points using scroll and delete
            # We use scroll to get all point IDs, then delete them
            scroll_result = self.qdrant_client.scroll(
                collection_name=name,
                limit=10000,  # Get up to 10k points at a time
                with_payload=False,  # We only need IDs
                with_vectors=False   # We don't need vectors
            )
            
            all_point_ids = []
            points, next_page_offset = scroll_result
            
            # Collect all point IDs
            while points:
                all_point_ids.extend([point.id for point in points])
                
                if next_page_offset is None:
                    break
                    
                # Get next batch
                scroll_result = self.qdrant_client.scroll(
                    collection_name=name,
                    offset=next_page_offset,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False
                )
                points, next_page_offset = scroll_result
            
            # Delete all points in batches
            if all_point_ids:
                batch_size = 1000  # Delete in smaller batches to avoid timeouts
                for i in tqdm(range(0, len(all_point_ids), batch_size), desc="Deleting points"):
                    batch_ids = all_point_ids[i:i + batch_size]
                    self.qdrant_client.delete(
                        collection_name=name,
                        points_selector=models.PointIdsList(points=batch_ids)
                    )
                
                print(f"✅ Successfully deleted all {len(all_point_ids)} points from collection '{name}'")
                return True
            else:
                print(f"No points found to delete in collection '{name}'")
                return True
                
        except Exception as e:
            print(f"❌ Error deleting data from collection '{name}': {e}")
            return False

    def delete_collection(self, name=None):
        """
        Completely deletes the entire collection including its structure.
        Use this if you want to remove the collection entirely.
        """
        name = name or self.collection_name
        try:
            if self.qdrant_client.collection_exists(name):
                self.qdrant_client.delete_collection(name)
                print(f"✅ Successfully deleted collection '{name}'")
                return True
            else:
                print(f"Collection '{name}' does not exist.")
                return False
        except Exception as e:
            print(f"❌ Error deleting collection '{name}': {e}")
            return False

    def get_collection_info(self, name=None):
        """
        Returns information about the collection including point count and configuration.
        """
        name = name or self.collection_name
        try:
            if not self.qdrant_client.collection_exists(name):
                return {"error": f"Collection '{name}' does not exist"}
            
            collection_info = self.qdrant_client.get_collection(name)
            return {
                "collection_name": name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": f"Error getting collection info: {e}"}
    
    def __del__(self):
        """Cleanup: Note that we don't auto-stop Qdrant as it may be used by other processes."""
        # We intentionally don't stop Qdrant here as it's typically a shared service
        # Users can manually call stop_qdrant_server() if needed
        pass

