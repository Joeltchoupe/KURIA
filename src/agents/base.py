"""
KURIA Agents — Base abstraite.

Contrat strict que chaque agent doit respecter :
1. _validate(data)      → Vérifie les données d'entrée, retourne warnings
2. _execute(data)       → Logique métier pure, retourne dict
3. _confidence(in, out) → Score de confiance 0.0-1.0
4. _emit_event(...)     → Collecte événements, flush après execute()

Le cycle de vie est géré par execute() qui orchestre tout :
  validate → execute → confidence → flush events → return AgentResult
"""

from __future__ import annotations

import abc
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from models.agent_config import BaseAgentConfig
from models.events import Event, EventType, EventPriority


# ══════════════════════════════════════════
# TYPES
# ══════════════════════════════════════════

ConfigT = TypeVar("ConfigT", bound=BaseAgentConfig)


class AgentResult(BaseModel):
    """Résultat standardisé de tout agent."""

    agent_name: str
    company_id: str
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution_time_ms: int = 0
    events_emitted: int = 0
    mode: str = "full"  # "full" | "observation" | "degraded"


# ══════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════


class AgentError(Exception):
    """Erreur générique agent."""

    def __init__(self, agent_name: str, detail: str):
        self.agent_name = agent_name
        self.detail = detail
        super().__init__(f"[{agent_name}] {detail}")


class InsufficientDataError(AgentError):
    """Pas assez de données pour analyser — déclenche le mode observation."""

    def __init__(self, agent_name: str, detail: str, available: int = 0, required: int = 0):
        self.available = available
        self.required = required
        super().__init__(agent_name=agent_name, detail=detail)


class ConnectorDataError(AgentError):
    """Données du connecteur invalides ou corrompues."""
    pass


# ══════════════════════════════════════════
# BASE AGENT — ABC
# ══════════════════════════════════════════


class BaseAgent(abc.ABC, Generic[ConfigT]):
    """Classe abstraite pour tous les agents KURIA.

    Cycle de vie :
        result = await agent.execute(data)

    Implémentation requise :
        _validate(data)         → list[str]  (warnings)
        _execute(data)          → dict[str, Any]
        _confidence(in, out)    → float
    """

    AGENT_NAME: str = "base"  # Override dans chaque sous-classe

    def __init__(
        self,
        company_id: str,
        config: ConfigT,
        *,
        run_id: Optional[str] = None,
        supabase_client: Any = None,
        claude_client: Any = None,
    ):
        self.company_id = company_id
        self.config = config
        self.run_id = run_id or str(uuid.uuid4())
        self.supabase = supabase_client
        self.claude = claude_client

        # Structured logger
        self.logger = logging.getLogger(f"kuria.agent.{self.AGENT_NAME}")
        self._log_context = {
            "agent": self.AGENT_NAME,
            "company_id": self.company_id,
            "run_id": self.run_id,
        }

        # Event buffer — flush après execute()
        self._pending_events: list[Event] = []

    # ──────────────────────────────────────
    # PUBLIC API — Point d'entrée unique
    # ──────────────────────────────────────

    async def execute(self, data: dict[str, Any]) -> AgentResult:
        """Orchestre le cycle complet : validate → execute → confidence → result.

        Ne PAS override cette méthode. Override _validate, _execute, _confidence.
        """
        self._pending_events.clear()
        start = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []
        mode = "full"
        output_data: dict[str, Any] = {}
        confidence = 0.0

        try:
            # 1. Validation
            self._log("info", "Starting validation")
            warnings = self._validate(data)
            if warnings:
                self._log("warning", f"Validation warnings: {warnings}")

            # 2. Exécution
            self._log("info", "Starting execution")
            output_data = await self._execute(data)
            mode = output_data.pop("__mode__", "full")

            # 3. Confidence
            confidence = self._confidence(data, output_data)
            self._log("info", f"Confidence: {confidence:.2f}, mode: {mode}")

        except InsufficientDataError as e:
            self._log("warning", f"Insufficient data: {e.detail}")
            mode = "observation"
            output_data = await self._observation_mode(data, e)
            confidence = self._confidence(data, output_data)
            warnings.append(e.detail)

        except ConnectorDataError as e:
            self._log("error", f"Connector error: {e.detail}")
            mode = "degraded"
            errors.append(e.detail)
            confidence = 0.0

        except Exception as e:
            self._log("error", f"Unexpected error: {e}", exc_info=True)
            errors.append(str(e))
            confidence = 0.0

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # 4. Flush events
        events_count = len(self._pending_events)
        await self._flush_events()

        # 5. Persist metrics
        await self._record_metrics(output_data, confidence, elapsed_ms)

        result = AgentResult(
            agent_name=self.AGENT_NAME,
            company_id=self.company_id,
            run_id=self.run_id,
            data=output_data,
            confidence=confidence,
            warnings=warnings,
            errors=errors,
            execution_time_ms=elapsed_ms,
            events_emitted=events_count,
            mode=mode,
        )

        self._log("info", f"Completed in {elapsed_ms}ms — {mode}")
        return result

    # ──────────────────────────────────────
    # ABSTRACT — À implémenter par chaque agent
    # ──────────────────────────────────────

    @abc.abstractmethod
    def _validate(self, data: dict[str, Any]) -> list[str]:
        """Valide les données d'entrée. Retourne la liste des warnings.
        Raise InsufficientDataError ou ConnectorDataError si bloquant."""
        ...

    @abc.abstractmethod
    async def _execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Logique métier pure. Retourne les résultats.
        Peut injecter __mode__ = 'observation' dans le retour."""
        ...

    @abc.abstractmethod
    def _confidence(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> float:
        """Calcule le score de confiance entre 0.0 et 1.0."""
        ...

    async def _observation_mode(self, data: dict[str, Any], error: InsufficientDataError) -> dict[str, Any]:
        """Mode dégradé par défaut. Override pour personnaliser."""
        return {
            "mode": "observation",
            "message": error.detail,
            "available": error.available,
            "required": error.required,
        }

    # ──────────────────────────────────────
    # EVENT SYSTEM
    # ──────────────────────────────────────

    def _emit_event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        priority: EventPriority = EventPriority.MEDIUM,
    ) -> None:
        """Ajoute un événement au buffer. Sera flush après execute()."""
        event = Event(
            event_type=event_type,
            priority=priority,
            agent_name=self.AGENT_NAME,
            company_id=self.company_id,
            run_id=self.run_id,
            payload=payload,
        )
        self._pending_events.append(event)
        self._log("debug", f"Event queued: {event_type.value}")

    async def _flush_events(self) -> None:
        """Persiste tous les événements en attente."""
        if not self._pending_events:
            return

        if self.supabase:
            try:
                events_data = [e.model_dump(mode="json") for e in self._pending_events]
                await self.supabase.insert_batch("agent_events", events_data)
            except Exception as e:
                self._log("error", f"Failed to flush {len(self._pending_events)} events: {e}")
        else:
            self._log("debug", f"No supabase client — {len(self._pending_events)} events logged only")
            for event in self._pending_events:
                self._log("info", f"EVENT: {event.event_type.value} | {event.payload}")

        self._pending_events.clear()

    # ──────────────────────────────────────
    # PERSISTENCE — Métriques
    # ──────────────────────────────────────

    async def _record_metrics(
        self,
        output_data: dict[str, Any],
        confidence: float,
        execution_time_ms: int,
    ) -> None:
        """Persiste les métriques de ce run."""
        if not self.supabase:
            return

        try:
            await self.supabase.insert("agent_runs", {
                "run_id": self.run_id,
                "agent_name": self.AGENT_NAME,
                "company_id": self.company_id,
                "confidence": confidence,
                "execution_time_ms": execution_time_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            self._log("error", f"Failed to record metrics: {e}")

    async def get_previous_metrics(self) -> Optional[dict[str, Any]]:
        """Récupère les métriques du run précédent pour comparaison."""
        if not self.supabase:
            return None

        try:
            return await self.supabase.get_latest(
                "agent_metrics",
                filters={
                    "agent_name": self.AGENT_NAME,
                    "company_id": self.company_id,
                },
                exclude_run_id=self.run_id,
            )
        except Exception:
            return None

    # ──────────────────────────────────────
    # HELPERS — Utilitaires partagés
    # ──────────────────────────────────────

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Division sûre."""
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def avg(values: list[float | int]) -> float:
        """Moyenne sûre."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def median(values: list[float | int]) -> float:
        """Médiane sûre."""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2 == 0:
            return (s[n // 2 - 1] + s[n // 2]) / 2
        return float(s[n // 2])

    @staticmethod
    def format_currency(amount: Optional[float], symbol: str = "€") -> str:
        """Formatte un montant."""
        if amount is None:
            return "N/A"
        return f"{amount:,.0f} {symbol}"

    @staticmethod
    def pct_change(current: float, previous: float) -> Optional[float]:
        """Variation en pourcentage."""
        if previous == 0:
            return None
        return round(((current - previous) / previous) * 100, 1)

    # ──────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────

    def _log(self, level: str, message: str, **kwargs) -> None:
        """Log structuré avec contexte agent."""
        log_fn = getattr(self.logger, level, self.logger.info)
        prefix = f"[{self.company_id}:{self.run_id[:8]}]"
        log_fn(f"{prefix} {message}", **kwargs)
