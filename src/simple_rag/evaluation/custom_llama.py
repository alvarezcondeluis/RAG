import json
import ollama
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# --- Imports for DeepEval Evaluation ---
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self, 
                 model_name: str = "llama3:8b",
                 faithfulness_threshold: float = 0.5,
                 answer_relevancy_threshold: float = 0.5,
                 contextual_relevancy_threshold: float = 0.5,
                 include_reason: bool = True):
        
        # This is the OLLAMA model name (e.g., "llama3:8b")
        self.model_name = model_name
        self.client = ollama.Client()
        self.async_client = ollama.AsyncClient()

        # --- Store evaluation parameters ---
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold
        self.contextual_relevancy_threshold = contextual_relevancy_threshold
        self.include_reason = include_reason
        self.test_cases: List[LLMTestCase] = []
        self.results = None

    def load_model(self):
        """This method is no longer needed as Ollama handles model loading."""
        return self.model_name

    def get_model_name(self) -> str:
        """Return the model name for DeepEval."""
        return self.model_name

    def _create_prompt_with_schema(self, prompt: str, schema: BaseModel) -> str:
        """Helper function to create a prompt that includes the schema definition."""
        schema_dict = schema.model_json_schema()
        schema_string = json.dumps(schema_dict, indent=2)

        return (
            f"{prompt}\n\n"
            "You MUST provide your response in a valid JSON format. "
            "Your response must strictly adhere to the following JSON schema:\n"
            f"```json\n{schema_string}\n```"
        )

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Generates a structured Pydantic object using Ollama."""
        full_prompt = self._create_prompt_with_schema(prompt, schema)
        schema_json = schema.model_json_schema()

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': full_prompt}
                ],
                format=schema_json
            )
            output = response['message']['content']
            json_result = json.loads(output)
            return schema(**json_result)
        
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            raise

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Asynchronously generates a structured Pydantic object using Ollama."""
        full_prompt = self._create_prompt_with_schema(prompt, schema)
        schema_json = schema.model_json_schema()
        
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'user', 'content': full_prompt}
                ],
                format=schema_json
            )
            output = response['message']['content']
            json_result = json.loads(output)
            return schema(**json_result)
        
        except Exception as e:
            print(f"Error during async Ollama generation: {e}")
            raise