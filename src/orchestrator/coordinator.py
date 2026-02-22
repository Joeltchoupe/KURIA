"""
AgentCoordinator — Signaux inter-agents.

Quand un agent détecte quelque chose qui concerne un autre agent,
il émet un Signal. Le Coordinator le route vers le bon destinataire.

Exemples :
  - Revenue Velocity détecte un deal stagnant depuis 30j
    → Signal vers Process Clarity : "vérifie le bottleneck"
  
  - Process Clarity détecte un cycle time × 2
    → Signal vers Cash Predictability : "recalcule le forecast"
  
  - Cash Predictability détecte un runway < 2 mois
    → Signal vers Revenue Velocity : "priorise les deals high value"

C'est ce qui fait que les 4 agents ne sont pas 4 silos.
C'est un SYSTÈME.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from models.agent_config import AgentType


# ══════════════════════════════════════════════════════════════
# SIGNAL
# ══════════════════════════════════════════════════════════════


class SignalType(str, Enum):
    """Type de signal inter-agent."""

    # Revenue Velocity → autres
    DEAL_STAGNANT = "deal_stagnant"
    PIPELINE_DROP = "pipeline_drop"
    HIGH_VALUE_AT_RISK = "high_value_at_risk"

    # Process Clarity → autres
    BOTTLENECK_DETECTED = "bottleneck_detected"
    CYCLE_TIME_SPIKE = "cycle_time_spike"
    HANDOFF_FAILURE = "handoff_failure"

    # Cash Predictability → autres
    RUNWAY_CRITICAL = "runway_critical"
    CASH_FORECAST_MISS = "cash_forecast_miss"
    PAYMENT_DELAY_DETECTED = "payment_delay_detected"

    # Acquisition Efficiency → autres
    CAC_SPIKE = "cac_spike"
    CHANNEL_DEGRADATION = "channel_degradation"
    LEAD_QUALITY_DROP = "lead_quality_drop"

    # Générique
    FRICTION_DETECTED = "friction_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    CONFIG_ADJUSTMENT_NEEDED = "config_adjustment_needed"


class SignalPriority(str, Enum):
    """Priorité d'un signal."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Signal(BaseModel):
    """
    Signal émis par un agent vers un ou plusieurs autres.

    Le Coordinator route le signal vers les destinataires
    et déclenche les actions appropriées.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    signal_type: SignalType
    priority: SignalPriority = SignalPriority.MEDIUM
    source_agent: AgentType
    target_agents: list[AgentType] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    emitted_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processed_at: datetime | None = None


# ══════════════════════════════════════════════════════════════
# ROUTING RULES
# ══════════════════════════════════════════════════════════════


# Quand un signal est émis, à qui le router ?
DEFAULT_ROUTING: dict[SignalType, list[AgentType]] = {
    # Revenue Velocity signals
    SignalType.DEAL_STAGNANT: [
        AgentType.PROCESS_CLARITY,
    ],
    SignalType.PIPELINE_DROP: [
        AgentType.CASH_PREDICTABILITY,
        AgentType.ACQUISITION_EFFICIENCY,
    ],
    SignalType.HIGH_VALUE_AT_RISK: [
        AgentType.CASH_PREDICTABILITY,
    ],

    # Process Clarity signals
    SignalType.BOTTLENECK_DETECTED: [
        AgentType.REVENUE_VELOCITY,
        AgentType.CASH_PREDICTABILITY,
    ],
    SignalType.CYCLE_TIME_SPIKE: [
        AgentType.CASH_PREDICTABILITY,
    ],
    SignalType.HANDOFF_FAILURE: [
        AgentType.REVENUE_VELOCITY,
    ],

    # Cash Predictability signals
    SignalType.RUNWAY_CRITICAL: [
        AgentType.REVENUE_VELOCITY,
        AgentType.ACQUISITION_EFFICIENCY,
    ],
    SignalType.CASH_FORECAST_MISS: [
        AgentType.REVENUE_VELOCITY,
    ],
    SignalType.PAYMENT_DELAY_DETECTED: [
        AgentType.PROCESS_CLARITY,
    ],

    # Acquisition Efficiency signals
    SignalType.CAC_SPIKE: [
        AgentType.CASH_PREDICTABILITY,
    ],
    SignalType.CHANNEL_DEGRADATION: [
        AgentType.REVENUE_VELOCITY,
    ],
    SignalType.LEAD_QUALITY_DROP: [
        AgentType.REVENUE_VELOCITY,
    ],

    # Génériques
    SignalType.FRICTION_DETECTED: [],  # Broadcast à tous
    SignalType.ANOMALY_DETECTED: [],
    SignalType.CONFIG_ADJUSTMENT_NEEDED: [],
}


# ══════════════════════════════════════════════════════════════
# COORDINATOR
# ══════════════════════════════════════════════════════════════


class AgentCoordinator:
    """
    Coordonne les interactions entre agents via des signaux.

    Les agents n'ont pas besoin de se connaître entre eux.
    Ils émettent des signaux, le Coordinator les route.

    Usage :
        coordinator = AgentCoordinator()

        # Un agent émet un signal
        coordinator.emit(Signal(
            signal_type=SignalType.DEAL_STAGNANT,
            source_agent=AgentType.REVENUE_VELOCITY,
            payload={"deal_id": "abc", "days_stagnant": 35},
            message="Deal 'Acme Corp' stagnant depuis 35 jours",
        ))

        # L'engine récupère les signaux en attente
        pending = coordinator.get_pending(target=AgentType.PROCESS_CLARITY)

        # Après traitement, flush
        processed = coordinator.flush_signals()
    """

    def __init__(
        self,
        routing: dict[SignalType, list[AgentType]] | None = None,
    ) -> None:
        self._routing = routing or DEFAULT_ROUTING
        self._signals: list[Signal] = []
        self._processed: list[Signal] = []

    # ──────────────────────────────────────────────────────
    # EMIT
    # ──────────────────────────────────────────────────────

    def emit(self, signal: Signal) -> Signal:
        """
        Émet un signal dans le système.

        Si target_agents n'est pas spécifié, utilise le routing par défaut.
        """
        if not signal.target_agents:
            signal.target_agents = self._routing.get(
                signal.signal_type, []
            )

        self._signals.append(signal)
        return signal

    def emit_friction(
        self,
        source_agent: AgentType,
        friction: Any,
        priority: SignalPriority = SignalPriority.MEDIUM,
    ) -> Signal:
        """Raccourci pour émettre un signal de friction détectée."""
        return self.emit(Signal(
            signal_type=SignalType.FRICTION_DETECTED,
            priority=priority,
            source_agent=source_agent,
            payload={
                "friction_type": getattr(friction, "type", "unknown"),
                "severity": getattr(friction, "severity", "unknown"),
                "title": getattr(friction, "title", "Friction détectée"),
                "estimated_cost": getattr(friction, "estimated_cost_monthly", 0),
            },
            message=getattr(friction, "description", ""),
        ))

    def process_frictions(
        self,
        source_agent: AgentType,
        frictions: list[Any],
    ) -> list[Signal]:
        """
        Analyse les frictions d'un agent et émet les signaux appropriés.

        Mappe les frictions vers les SignalTypes spécifiques
        quand c'est possible.
        """
        signals: list[Signal] = []

        for friction in frictions:
            title = getattr(friction, "title", "").lower()
            severity = getattr(friction, "severity", None)

            # Déterminer la priorité
            priority = SignalPriority.MEDIUM
            sev_value = severity.value if hasattr(severity, "value") else str(severity)
            if sev_value == "critical":
                priority = SignalPriority.CRITICAL
            elif sev_value == "high":
                priority = SignalPriority.HIGH

            # Mapper vers un signal spécifique si possible
            signal_type = self._friction_to_signal_type(source_agent, title)

            sig = self.emit(Signal(
                signal_type=signal_type,
                priority=priority,
                source_agent=source_agent,
                payload={
                    "friction_title": getattr(friction, "title", ""),
                    "friction_severity": sev_value,
                    "estimated_cost": getattr(friction, "estimated_cost_monthly", 0),
                },
                message=getattr(friction, "description", ""),
            ))
            signals.append(sig)

        return signals

    # ──────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────

    def get_pending(
        self,
        target: AgentType | None = None,
        priority: SignalPriority | None = None,
    ) -> list[Signal]:
        """
        Récupère les signaux en attente.

        Args:
            target: Filtrer par agent destinataire.
            priority: Filtrer par priorité minimum.
        """
        pending = [s for s in self._signals if not s.processed]

        if target is not None:
            pending = [
                s for s in pending
                if target in s.target_agents or not s.target_agents
            ]

        if priority is not None:
            priority_order = {
                SignalPriority.CRITICAL: 0,
                SignalPriority.HIGH: 1,
                SignalPriority.MEDIUM: 2,
                SignalPriority.LOW: 3,
            }
            max_level = priority_order[priority]
            pending = [
                s for s in pending
                if priority_order.get(s.priority, 3) <= max_level
            ]

        return sorted(pending, key=lambda s: s.emitted_at)

    def get_critical(self) -> list[Signal]:
        """Raccourci pour les signaux critiques non traités."""
        return self.get_pending(priority=SignalPriority.CRITICAL)

    @property
    def pending_count(self) -> int:
        """Nombre de signaux en attente."""
        return len([s for s in self._signals if not s.processed])

    # ──────────────────────────────────────────────────────
    # PROCESS / FLUSH
    # ──────────────────────────────────────────────────────

    def mark_processed(self, signal_id: str) -> None:
        """Marque un signal comme traité."""
        for s in self._signals:
            if s.id == signal_id:
                s.processed = True
                s.processed_at = datetime.utcnow()
                self._processed.append(s)
                return
        raise ValueError(f"Signal non trouvé : {signal_id}")

    def flush_signals(self) -> list[Signal]:
        """
        Marque tous les signaux en attente comme traités.

        Returns:
            Liste des signaux qui viennent d'être flushés.
        """
        flushed: list[Signal] = []
        for s in self._signals:
            if not s.processed:
                s.processed = True
                s.processed_at = datetime.utcnow()
                flushed.append(s)
                self._processed.append(s)
        return flushed

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    @property
    def total_signals(self) -> int:
        return len(self._signals)

    @property
    def total_processed(self) -> int:
        return len(self._processed)

    def get_signal_summary(self) -> dict[str, Any]:
        """Résumé des signaux pour le dashboard."""
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_priority: dict[str, int] = {}

        for s in self._signals:
            by_type[s.signal_type.value] = by_type.get(s.signal_type.value, 0) + 1
            by_source[s.source_agent.value] = by_source.get(s.source_agent.value, 0) + 1
            by_priority[s.priority.value] = by_priority.get(s.priority.value, 0) + 1

        return {
            "total": self.total_signals,
            "pending": self.pending_count,
            "processed": self.total_processed,
            "by_type": by_type,
            "by_source": by_source,
            "by_priority": by_priority,
        }

    # ──────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _friction_to_signal_type(
        source: AgentType, title: str
    ) -> SignalType:
        """Mappe un titre de friction vers un SignalType."""
        title_lower = title.lower()

        if source == AgentType.REVENUE_VELOCITY:
            if "stagnant" in title_lower or "zombie" in title_lower:
                return SignalType.DEAL_STAGNANT
            if "pipeline" in title_lower and "drop" in title_lower:
                return SignalType.PIPELINE_DROP
            if "high value" in title_lower or "risk" in title_lower:
                return SignalType.HIGH_VALUE_AT_RISK

        if source == AgentType.PROCESS_CLARITY:
            if "bottleneck" in title_lower:
                return SignalType.BOTTLENECK_DETECTED
            if "cycle time" in title_lower:
                return SignalType.CYCLE_TIME_SPIKE
            if "handoff" in title_lower:
                return SignalType.HANDOFF_FAILURE

        if source == AgentType.CASH_PREDICTABILITY:
            if "runway" in title_lower:
                return SignalType.RUNWAY_CRITICAL
            if "forecast" in title_lower:
                return SignalType.CASH_FORECAST_MISS
            if "payment" in title_lower or "retard" in title_lower:
                return SignalType.PAYMENT_DELAY_DETECTED

        if source == AgentType.ACQUISITION_EFFICIENCY:
            if "cac" in title_lower:
                return SignalType.CAC_SPIKE
            if "channel" in title_lower:
                return SignalType.CHANNEL_DEGRADATION
            if "lead quality" in title_lower:
                return SignalType.LEAD_QUALITY_DROP

        return SignalType.FRICTION_DETECTED
