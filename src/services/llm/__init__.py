"""
KURIA LLM — Client multi-fournisseur.

Supporte Claude (Anthropic), GPT (OpenAI), Gemini (Google).
L'agent choisit le provider, ou utilise le défaut.

3 modules :
  - client.py  → Connexion, retry, rate limiting, token tracking
  - prompts.py → Templates par agent, variables, versioning
  - parser.py  → Parsing des réponses, extraction JSON, validation
"""

from services.llm.client import LLMClient, LLMResponse
from services.llm.prompts import PromptManager, PromptTemplate
from services.llm.parser import ResponseParser

__all__ = [
    "LLMClient",
    "LLMResponse",
    "PromptManager",
    "PromptTemplate",
    "ResponseParser",
]
