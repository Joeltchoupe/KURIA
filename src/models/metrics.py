"""
Metrics — Métriques historiques de chaque agent.

Chaque agent sauvegarde ses métriques après chaque run.
L'Adaptateur les lit pour recalibrer.
Le dashboard les affiche pour montrer les tendances.

Design decisions :
- Un modèle de métriques par agent (pas de fourre-tout)
- Chaque métrique a un timestamp pour les séries temporelles
- MetricTrend pour les calculs de tendance standardisés
- Les métriques précédentes sont référencées pour le calcul de delta
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MetricTrend(BaseModel):
    """Tendance d'une métrique entre deux périodes."""

    current: float
    previous: Optional[float] = None
    delta: Optional[float] = None
    delta_pct: Optional[float] = None
    direction: str = Field(
        default="stable",
        description="'up', 'down', 'stable'",
    )

    @staticmethod
    def calculate(current: float, previous: Optional[float]) -> "MetricTrend":
        """Factory method pour calculer une tendance."""
        if previous is None or previous == 0:
            return MetricTrend(
                current=current,
                previous=previous,
                direction="stable",
            )

        delta = current - previous
        delta_pct = (delta / abs(previous)) * 100

        if abs(delta_pct) < 2:
            direction = "stable"
        elif delta > 0:
            direction = "up"
        else:
            direction = "down"

        return MetricTrend(
            current=current,
            previous=previous,
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 1),
            direction=direction,
        )

    @property
    def emoji(self) -> str:
        if self.direction == "up":
            return "↑"
        if self.direction == "down":
            return "↓"
        return "→"

    def format(self, unit: str = "", decimals: int = 0) -> str:
        """Format lisible pour le rapport."""
        current_str = f"{self.current:,.{decimals}f}{unit}"
        if self.delta_pct is not None:
            sign = "+" if self.delta_pct >= 0 else ""
            return f"{current_str} ({self.emoji}{sign}{self.delta_pct:.0f}%)"
        return current_str


class AgentMetrics(BaseModel):
    """Métriques de base communes à tous les agents."""

    company_id: str = Field(..., min_length=1)
    agent_type: str
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    run_duration_seconds: Optional[float] = Field(default=None, ge=0)
    tokens_used: Optional[int] = Field(default=None, ge=0)
    errors_count: int = Field(default=0, ge=0)
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Confiance globale de l'agent dans ses outputs pour ce run.",
    )


class RevenueVelocityMetrics(AgentMetrics):
    """Métriques de l'agent Revenue Velocity."""

    agent_type: str = "revenue_velocity"

    # ── KPI principal ──
    revenue_velocity_per_day: float = Field(
        ...,
        ge=0,
        description="€/jour traversant le pipeline",
    )

    # ── Pipeline ──
    pipeline_declared: float = Field(..., ge=0)
    pipeline_realistic: float = Field(..., ge=0)
    pipeline_gap_ratio: float = Field(..., ge=0)
    deals_active: int = Field(..., ge=0)
    deals_stagnant: int = Field(..., ge=0)
    deals_zombie: int = Field(..., ge=0)

    # ── Forecast ──
    forecast_30d: float = Field(..., ge=0)
    forecast_60d: float = Field(..., ge=0)
    forecast_90d: float = Field(..., ge=0)
    forecast_confidence: float = Field(..., ge=0, le=1.0)

    # ── Lead Scoring ──
    leads_scored: int = Field(default=0, ge=0)
    leads_hot: int = Field(default=0, ge=0)
    leads_archived: int = Field(default=0, ge=0)

    # ── Wins/Losses ──
    deals_won_this_period: int = Field(default=0, ge=0)
    deals_lost_this_period: int = Field(default=0, ge=0)
    revenue_won_this_period: float = Field(default=0, ge=0)


class ProcessClarityMetrics(AgentMetrics):
    """Métriques de l'agent Process Clarity."""

    agent_type: str = "process_clarity"

    # ── KPI principal ──
    avg_cycle_time_days: float = Field(
        ...,
        ge=0,
        description="Temps de cycle moyen des process critiques",
    )

    # ── Bottleneck ──
    bottleneck_stage: Optional[str] = None
    bottleneck_time_days: Optional[float] = Field(default=None, ge=0)
    bottleneck_estimated_cost: Optional[float] = Field(default=None, ge=0)

    # ── Tâches ──
    tasks_overdue: int = Field(default=0, ge=0)
    tasks_overdue_trend: str = Field(default="stable")

    # ── Waste ──
    waste_hours_per_week: Optional[float] = Field(default=None, ge=0)
    waste_annual_cost: Optional[float] = Field(default=None, ge=0)

    # ── Charge ──
    person_max_load_name: Optional[str] = None
    person_max_load_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
    )


class CashPredictabilityMetrics(AgentMetrics):
    """Métriques de l'agent Cash Predictability."""

    agent_type: str = "cash_predictability"

    # ── KPI principal ──
    forecast_accuracy_30d: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description=(
            "Précision du forecast à 30 jours. "
            "1.0 = parfait. None si pas encore assez d'historique."
        ),
    )

    # ── Cash position ──
    cash_position: float = Field(
        ...,
        description="Position cash actuelle (peut être négatif)",
    )
    runway_months: float = Field(..., ge=0)
    monthly_burn: float = Field(..., ge=0)

    # ── Forecast ──
    forecast_base_30d: float
    forecast_stress_30d: float
    forecast_upside_30d: float
    forecast_confidence: float = Field(..., ge=0, le=1.0)

    # ── Alertes ──
    alert_level: Optional[str] = Field(
        default=None,
        description="None, 'yellow', 'red'",
    )
    days_until_threshold: Optional[int] = Field(default=None, ge=0)

    # ── Recouvrement ──
    invoices_overdue_count: int = Field(default=0, ge=0)
    invoices_overdue_value: float = Field(default=0, ge=0)


class AcquisitionEfficiencyMetrics(AgentMetrics):
    """Métriques de l'agent Acquisition Efficiency."""

    agent_type: str = "acquisition_efficiency"

    # ── KPI principal ──
    blended_cac: Optional[float] = Field(
        default=None,
        ge=0,
        description="CAC blended. None si pas assez de données.",
    )

    # ── Par canal ──
    cac_by_channel: Optional[dict[str, float]] = None
    best_channel: Optional[str] = None
    worst_channel: Optional[str] = None

    # ── Volume ──
    total_leads: int = Field(default=0, ge=0)
    total_clients_acquired: int = Field(default=0, ge=0)

    # ── Qualité ──
    lead_to_client_conversion: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
    )
    avg_deal_size_by_channel: Optional[dict[str, float]] = None

    # ── Scoring canaux ──
    channels_performer: list[str] = Field(default_factory=list)
    channels_gouffre: list[str] = Field(default_factory=list)
    reallocation_recommendation: Optional[str] = None
