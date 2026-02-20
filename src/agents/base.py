"""
BaseAgent — Classe abstraite pour tous les agents Kuria.

C'est le contrat que chaque agent DOIT respecter.
C'est aussi la boîte à outils commune : appels LLM, events,
métriques, prompts, logging.

Design decisions :
- ABC strict avec run() comme point d'entrée unique
- AgentResult standardisé : chaque run produit le même type de sortie
- Appels LLM centralisés avec retry, cost tracking, structured output
- Publish/subscribe d'événements via le modèle Event
- Chargement de prompts depuis /prompts (pas de hardcode)
- Logging structuré avec context (company_id, agent_type)
- Métriques automatiques (durée, tokens, erreurs)

Architecture LLM :
- call_llm() pour du texte libre (narratifs, recommandations)
- call_llm_structured() pour du JSON validé (analyses, scores)
- Les deux utilisent le même client, le même retry, le même logging
- Le modèle est configurable par agent (Sonnet pour l'analyse, Haiku pour le routing)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from models.agent_config import BaseAgentConfig
from models.events import Event, EventType, EventPriority


# ──────────────────────────────────────────────
# EXCEPTIONS
# ──────────────────────────────────────────────


class AgentError(Exception):
    """Erreur générique d'un agent."""

    def __init__(
        self,
        agent_name: str,
        message: str,
        recoverable: bool = True,
        raw_error: Optional[Exception] = None,
    ):
        self.agent_name = agent_name
        self.message = message
        self.recoverable = recoverable
        self.raw_error = raw_error
        super().__init__(f"[Agent:{agent_name}] {message}")


class InsufficientDataError(AgentError):
    """Pas assez de données pour que l'agent soit fiable."""

    def __init__(self, agent_name: str, detail: str):
        super().__init__(
            agent_name=agent_name,
            message=f"Insufficient data: {detail}",
            recoverable=True,
        )


# ──────────────────────────────────────────────
# AGENT RESULT
# ──────────────────────────────────────────────

T = TypeVar("T", bound=BaseModel)


class AgentResult(BaseModel):
    """Résultat standardisé de chaque run d'agent.

    Chaque agent retourne ça. Pas un dict libre.
    Pas un type custom. AgentResult. Point.
    """

    agent_name: str
    company_id: str
    success: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Données produites (le type dépend de l'agent)
    data: Optional[dict[str, Any]] = None

    # Métriques du run
    duration_seconds: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    events_published: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Confiance
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Confiance globale dans les résultats de ce run.",
    )

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


# ──────────────────────────────────────────────
# LLM RESPONSE WRAPPER
# ──────────────────────────────────────────────


class LLMResponse(BaseModel):
    """Wrapper pour la réponse d'un appel LLM."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0


# ──────────────────────────────────────────────
# BASE AGENT
# ──────────────────────────────────────────────


class BaseAgent(ABC):
    """Classe abstraite pour tous les agents Kuria.

    Usage :
        agent = RevenueVelocityAgent(
            company_id="acme",
            config=revenue_config,
            llm_client=anthropic_client,
            db=database_service,
        )
        result = await agent.run(data=crm_snapshot)

    Chaque agent :
    1. Reçoit ses données via run(data=...)
    2. Analyse via des méthodes internes (LLM + logique)
    3. Publie des événements via publish_event()
    4. Retourne un AgentResult standardisé
    """

    AGENT_NAME: str = "base"
    PROMPTS_DIR: Path = Path(__file__).parent.parent / "prompts"

    def __init__(
        self,
        company_id: str,
        config: BaseAgentConfig,
        llm_client: Any,  # anthropic.AsyncAnthropic
        db: Any,  # services.database.DatabaseService
    ):
        self.company_id = company_id
        self.config = config
        self.llm_client = llm_client
        self.db = db

        self.logger = logging.getLogger(f"kuria.agents.{self.AGENT_NAME}")
        self._events_buffer: list[Event] = []
        self._total_tokens: int = 0
        self._total_llm_calls: int = 0
        self._run_start: Optional[float] = None

    # ── Point d'entrée principal ──

    async def run(self, data: dict[str, Any]) -> AgentResult:
        """Point d'entrée unique. Orchestre le cycle complet.

        1. Validation des données
        2. Exécution de l'analyse (implémentée par chaque agent)
        3. Publication des événements
        4. Sauvegarde des métriques
        5. Retour du résultat standardisé

        Args:
            data: Données normalisées depuis les connecteurs.

        Returns:
            AgentResult avec toutes les métriques.
        """
        self._run_start = time.time()
        self._total_tokens = 0
        self._total_llm_calls = 0
        self._events_buffer = []

        result = AgentResult(
            agent_name=self.AGENT_NAME,
            company_id=self.company_id,
            success=True,
        )

        try:
            # 1. Vérifier que l'agent est activé
            if not self.config.enabled:
                result.add_warning(f"Agent {self.AGENT_NAME} is disabled")
                result.success = True
                result.data = {}
                return result

            # 2. Valider les données d'entrée
            validation_errors = self._validate_input(data)
            if validation_errors:
                for err in validation_errors:
                    result.add_warning(f"Data validation: {err}")
                # On continue même avec des warnings — mode dégradé

            # 3. Exécuter l'analyse (méthode abstraite)
            analysis_data = await self.analyze(data)
            result.data = analysis_data

            # 4. Évaluer la confiance
            result.confidence = self._calculate_confidence(data, analysis_data)

            # 5. Si confiance trop basse, avertir
            if result.confidence < self.config.confidence_threshold:
                result.add_warning(
                    f"Confidence {result.confidence:.2f} below threshold "
                    f"{self.config.confidence_threshold}. Results may be unreliable."
                )

            # 6. Publier les événements bufferisés
            await self._flush_events()
            result.events_published = len(self._events_buffer)

        except InsufficientDataError as e:
            result.add_error(str(e))
            result.confidence = 0.0
            self.logger.warning(f"Insufficient data: {e}")

        except AgentError as e:
            result.add_error(str(e))
            self.logger.error(f"Agent error: {e}")
            if not e.recoverable:
                raise

        except Exception as e:
            result.add_error(f"Unexpected error: {e}")
            self.logger.exception(f"Unexpected error in {self.AGENT_NAME}")

        finally:
            # Métriques finales
            elapsed = time.time() - (self._run_start or time.time())
            result.duration_seconds = round(elapsed, 2)
            result.tokens_used = self._total_tokens
            result.llm_calls = self._total_llm_calls

            # Log
            self.logger.info(
                f"Run complete: success={result.success}, "
                f"confidence={result.confidence:.2f}, "
                f"duration={result.duration_seconds}s, "
                f"tokens={result.tokens_used}, "
                f"events={result.events_published}, "
                f"errors={len(result.errors)}"
            )

        return result

    # ── Méthode abstraite — chaque agent implémente ça ──

    @abstractmethod
    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse principale de l'agent.

        Args:
            data: Données normalisées.

        Returns:
            Dict avec les résultats d'analyse.
            La structure dépend de chaque agent.
        """
        ...

    @abstractmethod
    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        """Valide que les données d'entrée sont suffisantes.

        Returns:
            Liste d'erreurs/warnings. Liste vide = tout est bon.
        """
        ...

    @abstractmethod
    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        """Calcule le score de confiance pour ce run.

        Chaque agent définit ses propres critères de confiance.

        Returns:
            Score 0.0-1.0.
        """
        ...

    # ── LLM Calls ──

    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Appel LLM pour du texte libre (narratifs, recommandations).

        Utilise le modèle et les paramètres de la config agent
        sauf si override explicite.
        """
        model = model or self.config.llm_model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        start = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.llm_client.messages.create(**kwargs)

            elapsed = time.time() - start
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            self._total_tokens += input_tokens + output_tokens
            self._total_llm_calls += 1

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            self.logger.debug(
                f"LLM call: model={model}, "
                f"tokens={input_tokens}+{output_tokens}, "
                f"duration={elapsed:.2f}s"
            )

            return LLMResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                duration_seconds=round(elapsed, 2),
            )

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise AgentError(
                agent_name=self.AGENT_NAME,
                message=f"LLM call failed: {e}",
                recoverable=True,
                raw_error=e,
            )

    async def call_llm_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Appel LLM avec output JSON structuré.

        Le prompt DOIT demander du JSON.
        On parse et valide la réponse.

        Returns:
            Dict parsé depuis le JSON.

        Raises:
            AgentError si le JSON est invalide.
        """
        # Forcer température basse pour du JSON déterministe
        response = await self.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or self.config.llm_model,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        content = response.content.strip()

        # Extraire le JSON (il peut être dans un bloc ```json)
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            content = content[start:end].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse LLM JSON response: {e}\n"
                f"Raw content: {content[:500]}"
            )
            raise AgentError(
                agent_name=self.AGENT_NAME,
                message=f"LLM returned invalid JSON: {e}",
                recoverable=True,
                raw_error=e,
            )

    # ── Prompt Loading ──

    def load_prompt(self, template_name: str) -> str:
        """Charge un template de prompt depuis /prompts/{agent_name}/.

        Args:
            template_name: Nom du fichier (sans extension).

        Returns:
            Contenu du prompt.

        Raises:
            AgentError si le fichier n'existe pas.
        """
        prompt_path = self.PROMPTS_DIR / self.AGENT_NAME / f"{template_name}.txt"

        if not prompt_path.exists():
            # Fallback : chercher avec .md
            prompt_path = self.PROMPTS_DIR / self.AGENT_NAME / f"{template_name}.md"

        if not prompt_path.exists():
            raise AgentError(
                agent_name=self.AGENT_NAME,
                message=f"Prompt template not found: {prompt_path}",
                recoverable=False,
            )

        return prompt_path.read_text(encoding="utf-8")

    def render_prompt(
        self,
        template_name: str,
        **variables: Any,
    ) -> str:
        """Charge et rend un prompt avec des variables.

        Les variables sont remplacées par {variable_name} dans le template.
        """
        template = self.load_prompt(template_name)
        try:
            return template.format(**variables)
        except KeyError as e:
            raise AgentError(
                agent_name=self.AGENT_NAME,
                message=f"Missing variable in prompt template '{template_name}': {e}",
                recoverable=False,
            )

    # ── Event Publishing ──

    def publish_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        priority: Optional[EventPriority] = None,
    ) -> Event:
        """Publie un événement dans le buffer.

        Les événements sont flush vers la DB à la fin du run.
        Ça évite les écritures DB multiples pendant l'analyse.
        """
        event = Event(
            event_type=event_type,
            source_agent=self.AGENT_NAME,
            company_id=self.company_id,
            payload=payload,
            priority=priority or EventPriority.NORMAL,
        )
        self._events_buffer.append(event)

        self.logger.debug(
            f"Event buffered: {event_type.value} "
            f"(priority={event.priority.value})"
        )
        return event

    async def _flush_events(self) -> None:
        """Écrit tous les événements bufferisés dans la DB."""
        if not self._events_buffer:
            return

        try:
            for event in self._events_buffer:
                await self.db.save_event(event)

            self.logger.info(
                f"Flushed {len(self._events_buffer)} events to DB"
            )
        except Exception as e:
            self.logger.error(f"Failed to flush events: {e}")
            # On ne raise pas — les résultats sont valides même si
            # les événements ne sont pas persistés

    # ── Metrics & History ──

    async def get_previous_metrics(
        self,
        periods_back: int = 1,
    ) -> Optional[dict[str, Any]]:
        """Récupère les métriques du run précédent.

        Utilisé pour calculer les tendances et les deltas.
        """
        try:
            history = await self.db.get_metrics_history(
                company_id=self.company_id,
                agent_name=self.AGENT_NAME,
                limit=periods_back,
            )
            if history:
                return history[0]
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get previous metrics: {e}")
            return None

    async def save_metrics(self, metrics: BaseModel) -> None:
        """Sauvegarde les métriques de ce run."""
        try:
            await self.db.save_metrics(
                company_id=self.company_id,
                agent_name=self.AGENT_NAME,
                metrics=metrics.model_dump(),
            )
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    # ── Utility Methods ──

    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        default: float = 0.0,
    ) -> float:
        """Division sûre sans ZeroDivisionError."""
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def calculate_trend_pct(
        current: float,
        previous: Optional[float],
    ) -> Optional[float]:
        """Calcule le changement en pourcentage."""
        if previous is None or previous == 0:
            return None
        return round(((current - previous) / abs(previous)) * 100, 1)

    @staticmethod
    def avg(values: list[float]) -> float:
        """Moyenne sûre (retourne 0 si liste vide)."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def median(values: list[float]) -> float:
        """Médiane sûre."""
        if not values:
            return 0.0
        sorted_v = sorted(values)
        n = len(sorted_v)
        if n % 2 == 0:
            return (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
        return sorted_v[n // 2]

    @staticmethod
    def format_currency(amount: float, currency: str = "€") -> str:
        """Formate un montant en devise lisible."""
        if abs(amount) >= 1_000_000:
            return f"{amount / 1_000_000:,.1f}M{currency}"
        if abs(amount) >= 1_000:
            return f"{amount / 1_000:,.0f}K{currency}"
        return f"{amount:,.0f}{currency}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"company={self.company_id} "
            f"enabled={self.config.enabled}>"
)
