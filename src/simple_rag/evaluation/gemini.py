from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any
from pydantic import Field
from google import genai
from google.genai import types


class GeminiChatModel(BaseChatModel):
    """Custom LangChain wrapper for Google Gemini using new Google GenAI SDK."""
    
    model_name: str = Field(description="Name of the Gemini model to use")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    api_key: str = Field(description="Google API key")
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.0, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, temperature=temperature, **kwargs)
        # Create client with API key
        self.client = genai.Client(api_key=self.api_key)
    
    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Generate response from Gemini using new SDK."""
        # Convert messages to text prompt
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(msg.content)
        
        prompt = "\n\n".join(prompt_parts)
        
        # Generate response using new SDK
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                )
            )
            
            # Create ChatResult
            message = AIMessage(content=response.text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            # Handle blocked or failed responses
            error_message = AIMessage(content=f"Error generating response: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "temperature": self.temperature}