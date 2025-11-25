import time
from qdrant_client import models
from typing import Optional

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
        Searches Qdrant with an optional metadata filter.
        If the filter is provided and returns 0 results, it automatically 
        falls back to a normal (unfiltered) vector search.
        """
        query_embedding = self.embed_model.get_query_embedding(query)

        # --- First Attempt: Search with the provided filter ---
        start_time = time.time()
        results = self.vector_db.qdrant_client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            query_filter=metadata_filter,  # Pass the filter
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

        print(f"Retriever: Initial search took {elapsed_time:.4f} seconds.")
        if metadata_filter:
            print(f"Retriever: Search was filtered with: {metadata_filter}")

        # --- NEW: Fallback Logic ---
        # If a filter was used AND it returned no results, search again without it.
        if metadata_filter is not None and len(results) == 0:
            print("Retriever: Filter returned 0 results. Falling back to normal vector search...")
            
            start_time_fallback = time.time()
            results = self.vector_db.qdrant_client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                query_filter=None,  # <-- Run the search again with no filter
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
            end_time_fallback = time.time()
            elapsed_time_fallback = end_time_fallback - start_time_fallback
            print(f"Retriever: Fallback search took {elapsed_time_fallback:.4f} seconds.")
        
        return results