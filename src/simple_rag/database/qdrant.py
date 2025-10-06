from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct
from tqdm import tqdm

def batch_iterate(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class QdrantDatabase:
    def __init__(self, collection_name, vector_size=1024, batch_size=512):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.qdrant_client = QdrantClient(url="http://localhost:6334", prefer_grpc=True)

    def create_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                )
            )

    def batch_upsert(self, embeddings_list):
        """
        embeddings_list: List of dictionaries with structure:
        {
            'vector': {'text_vector': [...]},
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
                        vector=embedding_entry['vector']['text_vector'],  # Extract the actual vector
                        payload=embedding_entry['payload']  # Use the payload as-is
                    )
                )
                point_id += 1
            
            # Use upsert instead of upload_collection
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        # Update indexing threshold after all data is uploaded
        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

    def delete_all_data(self):
        """
        Deletes all data (points) from the collection while keeping the collection structure intact.
        This is useful for clearing the database without recreating the collection.
        """
        try:
            # Check if collection exists first
            if not self.qdrant_client.collection_exists(self.collection_name):
                print(f"Collection '{self.collection_name}' does not exist.")
                return False
            
            # Get collection info to check if it has any points
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            if points_count == 0:
                print(f"Collection '{self.collection_name}' is already empty.")
                return True
            
            print(f"Deleting {points_count} points from collection '{self.collection_name}'...")
            
            # Delete all points using scroll and delete
            # We use scroll to get all point IDs, then delete them
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
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
                    collection_name=self.collection_name,
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
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(points=batch_ids)
                    )
                
                print(f"✅ Successfully deleted all {len(all_point_ids)} points from collection '{self.collection_name}'")
                return True
            else:
                print(f"No points found to delete in collection '{self.collection_name}'")
                return True
                
        except Exception as e:
            print(f"❌ Error deleting data from collection '{self.collection_name}': {e}")
            return False

    def delete_collection(self):
        """
        Completely deletes the entire collection including its structure.
        Use this if you want to remove the collection entirely.
        """
        try:
            if self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.delete_collection(self.collection_name)
                print(f"✅ Successfully deleted collection '{self.collection_name}'")
                return True
            else:
                print(f"Collection '{self.collection_name}' does not exist.")
                return False
        except Exception as e:
            print(f"❌ Error deleting collection '{self.collection_name}': {e}")
            return False

    def get_collection_info(self):
        """
        Returns information about the collection including point count and configuration.
        """
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                return {"error": f"Collection '{self.collection_name}' does not exist"}
            
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": f"Error getting collection info: {e}"}

# Usage example:
# embeddings = []  # Your list of embedding dictionaries
# database = QdrantDatabase(collection_name="unstructured_parsing")
# database.create_collection()
# database.batch_upsert(embeddings)