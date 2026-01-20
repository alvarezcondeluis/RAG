
# Corrected code
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from typing import List
def batch_iterate(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

class EmbedData:
    def __init__(self, model_name, batch_size = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = self._load_model()
        self.embeddings = []

    def _load_model(self):
        model = HuggingFaceEmbedding(model_name=self.model_name,trust_remote_code=True, cache_folder="./cache")
        return model

    def generate_embedding(self, context):
        return self.model.get_text_embedding_batch(context)
        
    def embed(self, texts: List[str], description: str = "Embedding") -> List[List[float]]:
        """
        Embeds a list of texts with a progress bar and error handling.
        """
        if not texts:
            return []
            
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        # TQDM is now inside the method for granular visibility
        with tqdm(total=len(texts), desc=description, unit="doc") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                try:
                    # Actual Model Call
                    batch_embs = self.model.get_text_embedding_batch(batch)
                    embeddings.extend(batch_embs)
                except Exception as e:
                    print(f"❌ Error in batch {i}: {e}")
                    # Fallback: insert None or zero-vectors to keep alignment
                    embeddings.extend([None] * len(batch)) 
                
                pbar.update(len(batch))
        
        return embeddings