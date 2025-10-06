import time
from qdrant_client import models

class Retriever:
    
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata
        


    def search(self, query, limit=5):
        query_embedding = self.embeddata.model.get_query_embedding(query)

        start_time = time.time()
        results = self.vector_db.qdrant_client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
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

        print(f"Execution time for the search: {elapsed_time:.4f} seconds")

        return results
        
        
        