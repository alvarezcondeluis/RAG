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

# Usage example:
# embeddings = []  # Your list of embedding dictionaries
# database = QdrantDatabase(collection_name="unstructured_parsing")
# database.create_collection()
# database.batch_upsert(embeddings)