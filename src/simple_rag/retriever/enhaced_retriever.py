import time
from qdrant_client import models

class EnhancedRetriever:
    
    def __init__(self, vector_db, embed_model):
        self.vector_db = vector_db
        self.embed_model = embed_model
        

    def search(self, 
               query: str, 
               limit: int = 10, 
               metadata_filter: Optional[models.Filter] = None 
              ):
        """
        Searches Qdrant, now with an optional metadata filter.
        """
        query_embedding = self.embed_model.get_query_embedding(query)

        start_time = time.time()
        results = self.vector_db.qdrant_client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            query_filter=metadata_filter,  
            limit=limit,
            search_params=models.SearchParams(
                quantization = models.QuantizationSearchParams(
                    ignore = True,
                    rescore = True,
                    oversampling = 2.0
                )
            ), 
            timeout = 1000
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Retriever: Search took {elapsed_time:.4f} seconds.")
        if metadata_filter:
            print(f"Retriever: Search was filtered with: {metadata_filter}")

        return results
        
        
        