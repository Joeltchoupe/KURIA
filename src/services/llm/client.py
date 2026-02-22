"""
LLMClient — Client multi-fournisseur unifié.

UN client, TROIS providers, MÊME interface.

Usage :
    from services.llm import LLMClient

    client = LLMClient()

    # Utilise le provider par défaut
    response = await client.complete("Analyse ce pipeline...")

    # Force un provider
    response = await client.complete(
        prompt="...",
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
    )

    # Mode rapide (Haiku/Mini/Flash)
    response = await client.complete_fast("Classe ce deal...")

Design decisions :
  - Async par défaut (les appels LLM sont I/O bound)
  - Retry avec backoff exponentiel
  - Token tracking par appel
  - Fallback automatique si un provider échoue
  - Chaque appel retourne un LLMResponse typé
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from services.config import Settings, get_settings, LLMProvider


# ══════════════════════════════════════════════════════════════
# RESPONSE MODEL
# ══════════════════════════════════════════════════════════════


class LLMResponse(BaseModel):
    """Réponse unifiée de n'importe quel provider."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: LLMProvider
    model: str
    content: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    success: bool = True

    @property
    def cost_eur(self) -> float:
        """Coût approximatif en EUR."""
        return round(self.cost_usd * 0.92, 6)


# ══════════════════════════════════════════════════════════════
# PRICING (USD par 1M tokens)
# ══════════════════════════════════════════════════════════════

PRICING: dict[str, dict[str, float]] = {
    # Claude
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-20250514": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Gemini
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estime le coût USD d'un appel."""
    prices = PRICING.get(model, {"input": 1.0, "output": 5.0})
    cost = (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )
    return round(cost, 6)


# ══════════════════════════════════════════════════════════════
# CLIENT
# ══════════════════════════════════════════════════════════════


class LLMClient:
    """
    Client LLM multi-fournisseur unifié.

    Gère :
      - Connexion à Claude, OpenAI, Gemini
      - Retry avec backoff exponentiel
      - Fallback automatique entre providers
      - Tracking des tokens et coûts
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._clients: dict[LLMProvider, Any] = {}
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0
        self._call_history: list[LLMResponse] = []

        self._init_clients()

    def _init_clients(self) -> None:
        """Initialise les clients pour chaque provider configuré."""
        if self._settings.has_claude:
            try:
                import anthropic
                self._clients[LLMProvider.CLAUDE] = anthropic.AsyncAnthropic(
                    api_key=self._settings.claude_api_key,
                    timeout=self._settings.llm_timeout_seconds,
                )
            except ImportError:
                pass

        if self._settings.has_openai:
            try:
                import openai
                self._clients[LLMProvider.OPENAI] = openai.AsyncOpenAI(
                    api_key=self._settings.openai_api_key,
                    timeout=self._settings.llm_timeout_seconds,
                )
            except ImportError:
                pass

        if self._settings.has_gemini:
            try:
                import google.genai as genai
                self._clients[LLMProvider.GEMINI] = genai.Client(
                    api_key=self._settings.gemini_api_key,
                )
            except ImportError:
                pass

    @property
    def available_providers(self) -> list[LLMProvider]:
        """Providers initialisés et prêts."""
        return list(self._clients.keys())

    @property
    def stats(self) -> dict[str, Any]:
        """Statistiques d'utilisation."""
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "available_providers": [p.value for p in self.available_providers],
        }

    # ──────────────────────────────────────────────────────
    # MAIN API
    # ──────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fallback: bool = True,
    ) -> LLMResponse:
        """
        Appel LLM unifié.

        Args:
            prompt: Le prompt utilisateur.
            system: Le system prompt (optionnel).
            provider: Force un provider (sinon utilise le défaut).
            model: Force un modèle (sinon utilise le défaut du provider).
            temperature: Override la température.
            max_tokens: Override le max tokens.
            fallback: Si True, essaie un autre provider en cas d'échec.

        Returns:
            LLMResponse avec le contenu et les métriques.
        """
        provider = provider or self._settings.default_llm_provider
        model = model or self._settings.get_default_model(provider)
        temperature = temperature or self._settings.llm_default_temperature
        max_tokens = max_tokens or self._settings.llm_default_max_tokens

        # Essayer le provider principal
        response = await self._call_with_retry(
            provider=provider,
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Fallback si échec
        if not response.success and fallback:
            for alt_provider in self.available_providers:
                if alt_provider == provider:
                    continue
                alt_model = self._settings.get_default_model(alt_provider)
                response = await self._call_with_retry(
                    provider=alt_provider,
                    model=alt_model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.success:
                    break

        # Track
        self._call_count += 1
        self._total_tokens += response.total_tokens
        self._total_cost_usd += response.cost_usd
        self._call_history.append(response)

        return response

    async def complete_fast(
        self,
        prompt: str,
        system: str | None = None,
        provider: LLMProvider | None = None,
    ) -> LLMResponse:
        """
        Appel LLM avec le modèle rapide/économique.

        Utilise Haiku / GPT-4o-mini / Gemini Flash Lite.
        Pour : classification, routing, parsing simple.
        """
        provider = provider or self._settings.default_llm_provider
        model = self._settings.get_fast_model(provider)

        return await self.complete(
            prompt=prompt,
            system=system,
            provider=provider,
            model=model,
            temperature=0.1,
            max_tokens=1024,
        )

    async def complete_structured(
        self,
        prompt: str,
        system: str | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Appel LLM optimisé pour les réponses structurées (JSON).

        Ajoute des instructions de formatage au system prompt.
        """
        json_instruction = (
            "Réponds UNIQUEMENT en JSON valide. "
            "Pas de texte avant ou après le JSON. "
            "Pas de markdown. Pas de ```json```. "
            "Juste le JSON brut."
        )

        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction

        return await self.complete(
            prompt=prompt,
            system=full_system,
            provider=provider,
            model=model,
            temperature=0.1,
        )

    # ──────────────────────────────────────────────────────
    # PROVIDER-SPECIFIC CALLS
    # ──────────────────────────────────────────────────────

    async def _call_with_retry(
        self,
        provider: LLMProvider,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Appel avec retry et backoff exponentiel."""
        max_retries = self._settings.llm_max_retries
        delay = self._settings.llm_retry_delay_seconds

        last_error: str = ""

        for attempt in range(max_retries):
            try:
                start = time.monotonic()

                if provider == LLMProvider.CLAUDE:
                    response = await self._call_claude(
                        model, prompt, system, temperature, max_tokens
                    )
                elif provider == LLMProvider.OPENAI:
                    response = await self._call_openai(
                        model, prompt, system, temperature, max_tokens
                    )
                elif provider == LLMProvider.GEMINI:
                    response = await self._call_gemini(
                        model, prompt, system, temperature, max_tokens
                    )
                else:
                    raise ValueError(f"Provider non supporté : {provider}")

                elapsed = (time.monotonic() - start) * 1000
                response.latency_ms = round(elapsed, 1)

                return response

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))

        return LLMResponse(
            provider=provider,
            model=model,
            content="",
            success=False,
            error=f"Échec après {max_retries} tentatives. Dernière erreur : {last_error}",
        )

    async def _call_claude(
        self,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Appel Anthropic Claude."""
        client = self._clients.get(LLMProvider.CLAUDE)
        if client is None:
            raise ConnectionError("Client Claude non initialisé")

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        raw = await client.messages.create(**kwargs)

        content = ""
        if raw.content and len(raw.content) > 0:
            content = raw.content[0].text

        return LLMResponse(
            provider=LLMProvider.CLAUDE,
            model=model,
            content=content,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            cost_usd=estimate_cost(model, raw.usage.input_tokens, raw.usage.output_tokens),
            raw_response={"id": raw.id, "stop_reason": raw.stop_reason},
        )

    async def _call_openai(
        self,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Appel OpenAI GPT."""
        client = self._clients.get(LLMProvider.OPENAI)
        if client is None:
            raise ConnectionError("Client OpenAI non initialisé")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        raw = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = raw.choices[0] if raw.choices else None
        content = choice.message.content if choice else ""

        input_tokens = raw.usage.prompt_tokens if raw.usage else 0
        output_tokens = raw.usage.completion_tokens if raw.usage else 0

        return LLMResponse(
            provider=LLMProvider.OPENAI,
            model=model,
            content=content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
            raw_response={"id": raw.id, "finish_reason": choice.finish_reason if choice else None},
        )

    async def _call_gemini(
        self,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Appel Google Gemini."""
        client = self._clients.get(LLMProvider.GEMINI)
        if client is None:
            raise ConnectionError("Client Gemini non initialisé")

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        raw = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        content = raw.text if raw.text else ""

        input_tokens = getattr(raw.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(raw.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            provider=LLMProvider.GEMINI,
            model=model,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
            raw_response={"model": model},
        )

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retourne l'historique des appels récents."""
        return [
            {
                "id": r.id,
                "provider": r.provider.value,
                "model": r.model,
                "tokens": r.total_tokens,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "created_at": r.created_at.isoformat(),
            }
            for r in self._call_history[-limit:]
  ]
