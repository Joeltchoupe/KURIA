"""
ConfigAdapter — Recalibration hebdomadaire des configs.

Chaque semaine, l'Adaptateur compare :
  - Ce que les agents ont PRÉDIT (forecast, scores, alerts)
  - Ce qui s'est RÉELLEMENT passé (données)

Et ajuste les configs en conséquence.

C'est le FEEDBACK LOOP qui rend le système plus intelligent
avec le temps. Le moat se creuse.

Design decisions :
  - L'Adaptateur ne change JAMAIS les configs au-delà des bornes
  - Chaque ajustement est tracé (qui, quoi, pourquoi)
  - L'Adaptateur peut être désactivé par client
  - Le CEO reçoit un résumé des ajustements dans le rapport hebdo
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from models.agent_config import (
    AgentConfigSet,
    AgentType,
    BaseAgentConfig,
    RevenueVelocityConfig,
    ProcessClarityConfig,
    CashPredictabilityConfig,
    AcquisitionEfficiencyConfig,
)


# ══════════════════════════════════════════════════════════════
# MODÈLES
# ══════════════════════════════════════════════════════════════


class Adjustment(BaseModel):
    """Un ajustement de paramètre par l'Adaptateur."""
    agent_type: AgentType
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    confidence: float = Field(default=0.5, ge=0, le=1)
    applied: bool = True


class AdaptationResult(BaseModel):
    """Résultat d'un cycle d'adaptation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    adjustments: list[Adjustment] = Field(default_factory=list)
    skipped_agents: list[AgentType] = Field(default_factory=list)
    adapted_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def adjustment_count(self) -> int:
        return len(self.adjustments)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "adjustments": self.adjustment_count,
            "skipped": len(self.skipped_agents),
            "adapted_at": self.adapted_at.isoformat(),
            "details": [
                {
                    "agent": a.agent_type.value,
                    "param": a.parameter,
                    "old": a.old_value,
                    "new": a.new_value,
                    "reason": a.reason,
                }
                for a in self.adjustments
            ],
        }


# ══════════════════════════════════════════════════════════════
# ADAPTER
# ══════════════════════════════════════════════════════════════


class ConfigAdapter:
    """
    Recalibre les configs des agents chaque semaine.

    Usage :
        adapter = ConfigAdapter()
        result = adapter.adapt(
            config_set=config_set,
            run_result=run_result,
            actuals={"revenue_velocity": {"forecast_accuracy": 0.72}},
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        max_adjustment_pct: float = 0.20,
    ) -> None:
        """
        Args:
            enabled: Si False, l'adaptation est skippée.
            max_adjustment_pct: Ajustement max par paramètre par cycle (20%).
        """
        self.enabled = enabled
        self.max_adjustment_pct = max_adjustment_pct
        self._history: list[AdaptationResult] = []

    def adapt(
        self,
        config_set: AgentConfigSet,
        run_result: Any | None = None,
        actuals: dict[str, dict[str, Any]] | None = None,
    ) -> AdaptationResult:
        """
        Lance un cycle d'adaptation.

        Args:
            config_set: Les configs actuelles.
            run_result: Le dernier RunResult (KPIs, frictions).
            actuals: Données réelles pour comparer aux prédictions.

        Returns:
            AdaptationResult avec tous les ajustements.
        """
        result = AdaptationResult(company_id=config_set.company_id)

        if not self.enabled:
            result.skipped_agents = [at for at in AgentType if at != AgentType.SCANNER]
            self._history.append(result)
            return result

        actuals = actuals or {}

        # Adapter chaque agent
        for agent_type, config in [
            (AgentType.REVENUE_VELOCITY, config_set.revenue_velocity),
            (AgentType.PROCESS_CLARITY, config_set.process_clarity),
            (AgentType.CASH_PREDICTABILITY, config_set.cash_predictability),
            (AgentType.ACQUISITION_EFFICIENCY, config_set.acquisition_efficiency),
        ]:
            if not config.enabled:
                result.skipped_agents.append(agent_type)
                continue

            agent_actuals = actuals.get(agent_type.value, {})
            agent_run = self._extract_agent_run(run_result, agent_type)

            adjustments = self._adapt_agent(
                agent_type=agent_type,
                config=config,
                agent_run=agent_run,
                actuals=agent_actuals,
            )

            for adj in adjustments:
                if adj.applied:
                    self._apply_adjustment(config, adj)
                result.adjustments.append(adj)

            if adjustments:
                config.increment_adaptation()

        config_set.updated_at = datetime.utcnow()
        self._history.append(result)

        return result

    # ──────────────────────────────────────────────────────
    # ADAPTATION PAR AGENT
    # ──────────────────────────────────────────────────────

    def _adapt_agent(
        self,
        agent_type: AgentType,
        config: BaseAgentConfig,
        agent_run: dict[str, Any] | None,
        actuals: dict[str, Any],
    ) -> list[Adjustment]:
        """Dispatch l'adaptation vers la méthode spécifique."""
        dispatch = {
            AgentType.REVENUE_VELOCITY: self._adapt_revenue_velocity,
            AgentType.PROCESS_CLARITY: self._adapt_process_clarity,
            AgentType.CASH_PREDICTABILITY: self._adapt_cash_predictability,
            AgentType.ACQUISITION_EFFICIENCY: self._adapt_acquisition_efficiency,
        }

        handler = dispatch.get(agent_type)
        if handler is None:
            return []

        return handler(config, agent_run, actuals)

    def _adapt_revenue_velocity(
        self,
        config: BaseAgentConfig,
        agent_run: dict[str, Any] | None,
        actuals: dict[str, Any],
    ) -> list[Adjustment]:
        """Adapte Revenue Velocity."""
        adjustments: list[Adjustment] = []

        if not isinstance(config, RevenueVelocityConfig):
            return adjustments

        # Ajuster forecast_correction_factor si on a la précision réelle
        forecast_accuracy = actuals.get("forecast_accuracy")
        if forecast_accuracy is not None and forecast_accuracy < 0.85:
            # Le forecast était trop optimiste
            old = config.forecast_correction_factor
            new = self._clamp(
                old * forecast_accuracy,
                0.5,
                1.5,
                old,
            )
            if new != old:
                adjustments.append(Adjustment(
                    agent_type=AgentType.REVENUE_VELOCITY,
                    parameter="forecast_correction_factor",
                    old_value=old,
                    new_value=round(new, 3),
                    reason=(
                        f"Forecast accuracy = {forecast_accuracy:.0%}. "
                        f"Correction {old:.2f} → {new:.2f}"
                    ),
                    confidence=0.7,
                ))

        # Ajuster stagnation_threshold si trop de faux positifs
        stagnation_false_positive_rate = actuals.get("stagnation_false_positive_rate")
        if stagnation_false_positive_rate is not None and stagnation_false_positive_rate > 0.3:
            old = config.stagnation_threshold_days
            new = min(old + 7, 90)
            if new != old:
                adjustments.append(Adjustment(
                    agent_type=AgentType.REVENUE_VELOCITY,
                    parameter="stagnation_threshold_days",
                    old_value=old,
                    new_value=new,
                    reason=(
                        f"Taux de faux positifs stagnation = {stagnation_false_positive_rate:.0%}. "
                        f"Seuil relevé de {old}j à {new}j"
                    ),
                    confidence=0.6,
                ))

        return adjustments

    def _adapt_process_clarity(
        self,
        config: BaseAgentConfig,
        agent_run: dict[str, Any] | None,
        actuals: dict[str, Any],
    ) -> list[Adjustment]:
        """Adapte Process Clarity."""
        adjustments: list[Adjustment] = []

        if not isinstance(config, ProcessClarityConfig):
            return adjustments

        # Ajuster hourly_rate si le client fournit un taux réel
        real_hourly_rate = actuals.get("hourly_rate")
        if real_hourly_rate is not None:
            old = config.hourly_rate_estimate
            if abs(real_hourly_rate - old) > 5:
                adjustments.append(Adjustment(
                    agent_type=AgentType.PROCESS_CLARITY,
                    parameter="hourly_rate_estimate",
                    old_value=old,
                    new_value=float(real_hourly_rate),
                    reason=f"Taux horaire réel fourni : {real_hourly_rate}€/h",
                    confidence=0.95,
                ))

        # Ajuster duplicate thresholds si trop de faux doublons
        dup_false_positive_rate = actuals.get("duplicate_false_positive_rate")
        if dup_false_positive_rate is not None and dup_false_positive_rate > 0.2:
            old = config.duplicate_auto_merge_threshold
            new = min(old + 0.02, 1.0)
            if new != old:
                adjustments.append(Adjustment(
                    agent_type=AgentType.PROCESS_CLARITY,
                    parameter="duplicate_auto_merge_threshold",
                    old_value=old,
                    new_value=round(new, 3),
                    reason=(
                        f"Taux de faux doublons = {dup_false_positive_rate:.0%}. "
                        f"Seuil auto-merge relevé à {new:.2f}"
                    ),
                    confidence=0.6,
                ))

        return adjustments

    def _adapt_cash_predictability(
        self,
        config: BaseAgentConfig,
        agent_run: dict[str, Any] | None,
        actuals: dict[str, Any],
    ) -> list[Adjustment]:
        """Adapte Cash Predictability."""
        adjustments: list[Adjustment] = []

        if not isinstance(config, CashPredictabilityConfig):
            return adjustments

        # Ajuster forecast_optimism_correction
        forecast_accuracy = actuals.get("forecast_accuracy")
        if forecast_accuracy is not None and forecast_accuracy < 0.80:
            old = config.forecast_optimism_correction
            new = self._clamp(
                old * forecast_accuracy,
                0.5,
                1.5,
                old,
            )
            if new != old:
                adjustments.append(Adjustment(
                    agent_type=AgentType.CASH_PREDICTABILITY,
                    parameter="forecast_optimism_correction",
                    old_value=old,
                    new_value=round(new, 3),
                    reason=(
                        f"Cash forecast accuracy = {forecast_accuracy:.0%}. "
                        f"Correction {old:.2f} → {new:.2f}"
                    ),
                    confidence=0.7,
                ))

        # Ajuster stress_payment_delay si les retards réels sont connus
        actual_avg_delay = actuals.get("avg_payment_delay_days")
        if actual_avg_delay is not None:
            old = config.stress_payment_delay_days
            if actual_avg_delay > old:
                new = min(int(actual_avg_delay * 1.2), 45)
                adjustments.append(Adjustment(
                    agent_type=AgentType.CASH_PREDICTABILITY,
                    parameter="stress_payment_delay_days",
                    old_value=old,
                    new_value=new,
                    reason=(
                        f"Retard moyen réel = {actual_avg_delay}j "
                        f"(stress scenario : {old}j → {new}j)"
                    ),
                    confidence=0.8,
                ))

        return adjustments

    def _adapt_acquisition_efficiency(
        self,
        config: BaseAgentConfig,
        agent_run: dict[str, Any] | None,
        actuals: dict[str, Any],
    ) -> list[Adjustment]:
        """Adapte Acquisition Efficiency."""
        adjustments: list[Adjustment] = []

        if not isinstance(config, AcquisitionEfficiencyConfig):
            return adjustments

        # Ajuster min_clients_for_analysis si le volume est connu
        total_clients = actuals.get("total_clients_signed")
        if total_clients is not None and total_clients < config.min_clients_for_analysis:
            adjustments.append(Adjustment(
                agent_type=AgentType.ACQUISITION_EFFICIENCY,
                parameter="enabled",
                old_value=config.enabled,
                new_value=False,
                reason=(
                    f"Seulement {total_clients} clients signés "
                    f"(minimum : {config.min_clients_for_analysis}). "
                    f"Agent désactivé — données insuffisantes."
                ),
                confidence=0.9,
                applied=True,
            ))

        return adjustments

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────

    def _apply_adjustment(
        self, config: BaseAgentConfig, adj: Adjustment
    ) -> None:
        """Applique un ajustement sur la config."""
        if hasattr(config, adj.parameter):
            setattr(config, adj.parameter, adj.new_value)

    def _clamp(
        self,
        value: float,
        min_val: float,
        max_val: float,
        current: float,
    ) -> float:
        """
        Clamp une valeur dans les bornes ET dans le max_adjustment_pct.

        L'ajustement ne peut pas dépasser max_adjustment_pct du current.
        """
        # Limiter l'ajustement à max_adjustment_pct
        max_delta = abs(current) * self.max_adjustment_pct
        clamped = max(current - max_delta, min(current + max_delta, value))

        # Puis borner
        return max(min_val, min(max_val, clamped))

    @staticmethod
    def _extract_agent_run(
        run_result: Any, agent_type: AgentType
    ) -> dict[str, Any] | None:
        """Extrait le run d'un agent spécifique depuis un RunResult."""
        if run_result is None:
            return None

        agent_runs = getattr(run_result, "agent_runs", [])
        for record in agent_runs:
            if getattr(record, "agent_type", None) == agent_type:
                return {
                    "status": getattr(record, "status", None),
                    "kpi_value": getattr(record, "kpi_value", None),
                    "frictions_detected": getattr(record, "frictions_detected", 0),
                    "result": getattr(record, "result", {}),
                }
        return None

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    @property
    def last_adaptation(self) -> AdaptationResult | None:
        return self._history[-1] if self._history else None

    @property
    def total_adaptations(self) -> int:
        return len(self._history)

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retourne l'historique des adaptations."""
        return [r.summary for r in self._history[-limit:]]
