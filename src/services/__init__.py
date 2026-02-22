"""
KURIA Services — Clients partagés.

UN client, UNE config, partagé partout.
Aucun module ne crée sa propre connexion.

Modules :
  - config.py          → Settings centralisés (.env → Pydantic)
  - llm/client.py      → Multi-LLM client (Claude, GPT, Gemini)
  - llm/prompts.py     → Template manager
  - llm/parser.py      → Extraction structurée des réponses
  - supabase.py        → Client Supabase + CRUD typés
  - notifications.py   → Resend (email) + Slack (webhook)
"""

from services.config import Settings, get_settings
from services.llm import LLMClient, PromptManager, ResponseParser
from services.supabase import SupabaseClient
from services.notifications import NotificationService

__all__ = [
    "Settings",
    "get_settings",
    "LLMClient",
    "PromptManager",
    "ResponseParser",
    "SupabaseClient",
    "NotificationService",
]
