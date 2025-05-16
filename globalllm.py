from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from langchain_groq import ChatGroq
from typing import List, Optional, Any
from pydantic import Field

class GroqLLM(LLM):
    model: str = Field(...)
    api_key: str = Field(...)
    temperature: float = Field(default=0.0)

    def _llm_type(self) -> str:
        return "groq-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        llm = ChatGroq(
            model = self.model,
            api_key = self.api_key,
            temperature = self.temperature,
            max_tokens = None,
            timeout = None,
            max_retries = 2,
        )
        return llm.invoke(prompt)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop)
            generations.append([Generation(text = output)])
        return LLMResult(generations=generations)
