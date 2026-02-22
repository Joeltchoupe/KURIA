"""
Settings — Configuration centralisée.

Toutes les variables d'environnement sont validées ICI.
Aucun os.getenv() ailleurs dans le code.

Usage :
    from services.config import get_settings

    settings = get_settings()
    settings.claude_api_key
    settings.supabase_url
    settings.default_llm_provider
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Fournisseurs LLM supportés."""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


class Environment(str, Enum):
    """Environnement d'exécution."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Configuration centralisée de Kuria.

    Charge depuis .env ou variables d'environnement.
    Chaque champ a une valeur par défaut raisonnable pour le dev.
    """

    # ── Environnement ──
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    app_name: str = "kuria"
    app_version: str = "0.1.0"

    # ── LLM : Claude ──
    claude_api_key: str = Field(
        default="",
        description="Clé API Anthropic",
    )
    claude_default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Modèle Claude par défaut (Sonnet pour l'analyse)",
    )
    claude_fast_model: str = Field(
        default="claude-haiku-4-20250514",
        description="Modèle Claude rapide (Haiku pour le routing/parsing)",
    )

    # ── LLM : OpenAI ──
    openai_api_key: str = Field(
        default="",
        description="Clé API OpenAI",
    )
    openai_default_model: str = Field(
        default="gpt-4o",
        description="Modèle OpenAI par défaut",
    )
    openai_fast_model: str = Field(
        default="gpt-4o-mini",
        description="Modèle OpenAI rapide",
    )

    # ── LLM : Gemini ──
    gemini_api_key: str = Field(
        default="",
        description="Clé API Google Gemini",
    )
    gemini_default_model: str = Field(
        default="gemini-2.5-flash",
        description="Modèle Gemini par défaut",
    )
    gemini_fast_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="Modèle Gemini rapide",
    )

    # ── LLM : Global ──
    default_llm_provider: LLMProvider = Field(
        default=LLMProvider.CLAUDE,
        description="Fournisseur LLM par défaut",
    )
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    llm_retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    llm_timeout_seconds: float = Field(default=120.0, ge=10, le=600)
    llm_default_temperature: float = Field(default=0.3, ge=0, le=1.0)
    llm_default_max_tokens: int = Field(default=4096, ge=256, le=32768)

    # ── Supabase ──
    supabase_url: str = Field(
        default="",
        description="URL du projet Supabase",
    )
    supabase_anon_key: str = Field(
        default="",
        description="Clé anonyme Supabase (publique)",
    )
    supabase_service_key: str = Field(
        default="",
        description="Clé service Supabase (privée, accès complet)",
    )

    # ── Notifications : Email (Resend) ──
    resend_api_key: str = Field(default="")
    resend_from_email: str = Field(
        default="kuria@notifications.kuria.ai",
        description="Adresse d'envoi des emails",
    )
    resend_reply_to: str = Field(
        default="support@kuria.ai",
    )

    # ── Notifications : Slack ──
    slack_webhook_url: str = Field(default="")
    slack_default_channel: str = Field(default="#kuria-alerts")

    # ── Connecteurs ──
    hubspot_api_key: str = Field(default="")
    quickbooks_client_id: str = Field(default="")
    quickbooks_client_secret: str = Field(default="")
    quickbooks_refresh_token: str = Field(default="")

    # ── Gmail ──
    gmail_credentials_json: str = Field(
        default="",
        description="Chemin vers le fichier credentials.json Google",
    )

    # ── API ──
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = Field(
        default="",
        description="Clé API pour authentifier les appels (n8n, dashboard)",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501"],
        description="Origines autorisées pour CORS (Streamlit par défaut)",
    )

    # ── Prompts ──
    prompts_dir: str = Field(
        default="prompts",
        description="Répertoire racine des templates de prompts",
    )

    # ── Logging ──
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # ── Validators ──

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v_upper

    # ── Properties ──

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def has_claude(self) -> bool:
        return bool(self.claude_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def available_providers(self) -> list[LLMProvider]:
        """Fournisseurs LLM configurés (avec clé API)."""
        providers = []
        if self.has_claude:
            providers.append(LLMProvider.CLAUDE)
        if self.has_openai:
            providers.append(LLMProvider.OPENAI)
        if self.has_gemini:
            providers.append(LLMProvider.GEMINI)
        return providers

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def has_resend(self) -> bool:
        return bool(self.resend_api_key)

    @property
    def has_slack(self) -> bool:
        return bool(self.slack_webhook_url)

    @property
    def has_hubspot(self) -> bool:
        return bool(self.hubspot_api_key)

    def get_llm_api_key(self, provider: LLMProvider) -> str:
        """Récupère la clé API pour un fournisseur donné."""
        mapping = {
            LLMProvider.CLAUDE: self.claude_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
        }
        return mapping.get(provider, "")

    def get_default_model(self, provider: LLMProvider) -> str:
        """Récupère le modèle par défaut pour un fournisseur."""
        mapping = {
            LLMProvider.CLAUDE: self.claude_default_model,
            LLMProvider.OPENAI: self.openai_default_model,
            LLMProvider.GEMINI: self.gemini_default_model,
        }
        return mapping.get(provider, "")

    def get_fast_model(self, provider: LLMProvider) -> str:
        """Récupère le modèle rapide pour un fournisseur."""
        mapping = {
            LLMProvider.CLAUDE: self.claude_fast_model,
            LLMProvider.OPENAI: self.openai_fast_model,
            LLMProvider.GEMINI: self.gemini_fast_model,
        }
        return mapping.get(provider, "")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton des settings.

    Chargé une seule fois, mis en cache.
    Usage : from services.config import get_settings
    """
    return Settings()
