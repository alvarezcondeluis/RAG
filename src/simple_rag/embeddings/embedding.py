
# Corrected code
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

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
        
    def embed(self, contexts):
        self.contexts = contexts
        self.embeddings = []
        
        for batch_context in tqdm(batch_iterate(contexts, self.batch_size),
                                  total=len(contexts)//self.batch_size,
                                  desc="Embedding data in batches"):
                                  
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)
        
        return self.embeddings