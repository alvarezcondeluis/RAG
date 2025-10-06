from llama_index.llms.ollama import Ollama

class RAG:
    def __init__(self, retriever, llm_name):
        self.retriever = retriever
        self.llm_name = llm_name
        self.llm = self._load_llm()
        self.prompt_tmpl = """
Context information is below.
---------------------
{context}
---------------------

Given the context information above I want you
to think step by step to answer the query in a
crisp manner, incase case you don't know the
answer say 'I don't know!' Do not say that you are using the context,
just answer more or less with a summary using the information of the context.

---------------------
Query: {query}
---------------------
Answer:
        """

    def _load_llm(self):
        try:
            llm = Ollama(model=self.llm_name, request_timeout=120.0)
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return None
        print("LLM loaded successfully")
        return llm
        

    def generate_context(self, query):
        results = self.retriever.search(query)
        combined_context = []             
        
        for result in results:
            # Extract text from the payload based on your Qdrant structure
            
           
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '')
                source = result.payload.get('source_document', 'Unknown')
                page = result.payload.get('page_number', 'N/A')
                section = result.payload.get('section_title', '')
                score = result.score
                
                print(f"Section: {section} score: {score}")
                
                # Format the context with metadata
                context_entry = f"[Source: {source}, Page: {page}]"
                if section:
                    context_entry += f" [Section: {section}]"
                context_entry += f"\n{text}"
                
                combined_context.append(context_entry)
        
        return "\n\n".join(combined_context)
    
    def query(self, question):
        # Generate context from retrieval
        context = self.generate_context(question)
        
        # Format the prompt
        formatted_prompt = self.prompt_tmpl.format(
            context=context,
            query=question
        )
        
        # Generate response using LLM
        response = self.llm.complete(formatted_prompt)
        
        return {
            'answer': response.text,
            'context': context
        }