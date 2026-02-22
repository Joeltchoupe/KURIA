"""
AgentConfig — Configuration de chaque agent par client.

Généré par l'orchestrateur (ClientProfileGenerator) après le scan.
Recalibré chaque semaine par l'Adaptateur.

Chaque config contient les SEUILS et PARAMÈTRES spécifiques
au contexte de l'entreprise. C'est ce qui fait que le même agent
se comporte différemment pour chaque client.

C'est LE MOAT : l'orchestration contextuelle.

Design decisions :
- BaseAgentConfig = socle commun (LLM, fréquence, alertes, confiance)
- Configs spécifiques = typage fort par agent (seuils métier bornés)
- Threshold = modèle générique pour les seuils d'alerte
- AgentConfig = format générique que les agents consomment
  (expose .parameters dict + .thresholds list)
- Les configs spécifiques GÉNÈRENT un AgentConfig via .to_agent_config()
- Compatibilité bidirectionnelle : typage fort pour l'orchestrateur,
  format plat pour les agents
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════


class AgentType(str, Enum):
    """Les 4 agents + le scanner."""

    SCANNER = "scanner"
    REVENUE_VELOCITY = "revenue_velocity"
    PROCESS_CLARITY = "process_clarity"
    CASH_PREDICTABILITY = "cash_predictability"
    ACQUISITION_EFFICIENCY = "acquisition_efficiency"


# ══════════════════════════════════════════════════════════════
# THRESHOLD — Format générique pour les seuils d'alerte
# ══════════════════════════════════════════════════════════════


class Threshold(BaseModel):
    """
    Seuil d'alerte générique.

    Utilisé par les agents pour évaluer la santé d'une métrique.
    Direction :
      - "above" = alerte si valeur > seuil (ex: cycle_time trop long)
      - "below" = alerte si valeur < seuil (ex: on_time_rate trop bas)
    """

    metric_name: str = Field(..., min_length=1)
    warning_value: float
    critical_value: float
    direction: str = Field(default="above")

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        allowed = {"above", "below"}
        if v not in allowed:
            raise ValueError(f"direction must be one of {allowed}")
        return v

    def evaluate(self, value: float) -> str:
        """
        Évalue une valeur contre ce seuil.

        Returns:
            'critical', 'warning', ou 'healthy'
        """
        if self.direction == "above":
            if value >= self.critical_value:
                return "critical"
            if value >= self.warning_value:
                return "warning"
            return "healthy"
        else:  # below
            if value <= self.critical_value:
                return "critical"
            if value <= self.warning_value:
                return "warning"
            return "healthy"


# ══════════════════════════════════════════════════════════════
# AGENT CONFIG — Format générique consommé par les agents
# ══════════════════════════════════════════════════════════════


class AgentConfig(BaseModel):
    """
    Configuration générique consommée par les agents.

    C'est le format que chaque agent reçoit dans son __init__.
    Les configs spécifiques (RevenueVelocityConfig, etc.) génèrent
    ce format via .to_agent_config().

    On peut aussi créer un AgentConfig directement pour les tests
    ou les cas simples.
    """

    agent_type: AgentType
    enabled: bool = True
    company_id: str = Field(default="", description="Vide pour les tests")

    # ── Seuils ──
    thresholds: list[Threshold] = Field(default_factory=list)

    # ── Paramètres plats ──
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres clé-valeur consommés par l'agent",
    )

    # ── Scheduling ──
    schedule_cron: str = Field(
        default="0 6 * * 1",
        description="Cron expression pour l'exécution planifiée",
    )

    # ── Sources de données ──
    data_sources: list[str] = Field(
        default_factory=list,
        description="Connecteurs utilisés par cet agent",
    )

    # ── LLM ──
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    temperature: float = Field(default=0.3, ge=0, le=1.0)

    # ── Notifications ──
    alert_channel: str = Field(default="email")
    alert_recipients: list[str] = Field(default_factory=list)

    # ── Confiance ──
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)

    # ── Méta ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    adaptation_count: int = Field(default=0, ge=0)

    def get_threshold(self, metric_name: str) -> Threshold | None:
        """Récupère un threshold par nom de métrique."""
        for t in self.thresholds:
            if t.metric_name == metric_name:
                return t
        return None

    def increment_adaptation(self) -> None:
        """Appelé par l'Adaptateur après chaque recalibration."""
        self.adaptation_count += 1
        self.updated_at = datetime.utcnow()


# ══════════════════════════════════════════════════════════════
# BASE AGENT CONFIG — Socle commun typé fort
# ══════════════════════════════════════════════════════════════


class BaseAgentConfig(BaseModel):
    """
    Configuration commune à tous les agents (version typée forte).

    Utilisée par l'orchestrateur pour la calibration.
    Chaque config spécifique hérite de celle-ci et expose
    .to_agent_config() pour générer le format plat.
    """

    agent_type: AgentType
    enabled: bool = True
    company_id: str = Field(..., min_length=1)

    # ── LLM ──
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Modèle Claude à utiliser.",
    )
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    temperature: float = Field(default=0.3, ge=0, le=1.0)

    # ── Fréquence ──
    run_frequency: str = Field(
        default="daily",
        description="'realtime', 'hourly', 'daily', 'weekly', 'monthly'",
    )

    # ── Notifications ──
    alert_channel: str = Field(default="email")
    alert_recipients: list[str] = Field(default_factory=list)

    # ── Confiance ──
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)

    # ── Méta ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    adaptation_count: int = Field(default=0, ge=0)

    @field_validator("run_frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        allowed = {"realtime", "hourly", "daily", "weekly", "monthly"}
        if v not in allowed:
            raise ValueError(f"run_frequency must be one of {allowed}")
        return v

    @field_validator("alert_channel")
    @classmethod
    def validate_alert_channel(cls, v: str) -> str:
        allowed = {"email", "slack", "both"}
        if v not in allowed:
            raise ValueError(f"alert_channel must be one of {allowed}")
        return v

    def increment_adaptation(self) -> None:
        """Appelé par l'Adaptateur après chaque recalibration."""
        self.adaptation_count += 1
        self.updated_at = datetime.utcnow()

    def _frequency_to_cron(self) -> str:
        """Convertit run_frequency en cron expression."""
        mapping = {
            "realtime": "* * * * *",
            "hourly": "0 * * * *",
            "daily": "0 6 * * *",
            "weekly": "0 6 * * 1",
            "monthly": "0 6 1 * *",
        }
        return mapping.get(self.run_frequency, "0 6 * * 1")

    def _base_to_agent_config_kwargs(self) -> dict[str, Any]:
        """Champs de base pour construire un AgentConfig."""
        return {
            "agent_type": self.agent_type,
            "enabled": self.enabled,
            "company_id": self.company_id,
            "llm_model": self.llm_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "alert_channel": self.alert_channel,
            "alert_recipients": self.alert_recipients,
            "confidence_threshold": self.confidence_threshold,
            "schedule_cron": self._frequency_to_cron(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "adaptation_count": self.adaptation_count,
        }

    def to_agent_config(self) -> AgentConfig:
        """
        Convertit cette config typée en AgentConfig générique.

        Doit être surchargée par chaque config spécifique
        pour injecter thresholds + parameters.
        """
        return AgentConfig(
            **self._base_to_agent_config_kwargs(),
            thresholds=[],
            parameters={},
            data_sources=[],
        )


# ══════════════════════════════════════════════════════════════
# CONFIGS SPÉCIFIQUES
# ══════════════════════════════════════════════════════════════


class ScoreWeights(BaseModel):
    """Poids du lead scoring. Doivent sommer à 1.0."""

    fit: float = Field(default=0.3, ge=0, le=1.0)
    activity: float = Field(default=0.4, ge=0, le=1.0)
    timing: float = Field(default=0.2, ge=0, le=1.0)
    source: float = Field(default=0.1, ge=0, le=1.0)

    @field_validator("source")
    @classmethod
    def validate_sum(cls, v, info):
        """Vérifie que les poids somment à 1.0 (tolérance 0.01)."""
        data = info.data
        total = (
            data.get("fit", 0)
            + data.get("activity", 0)
            + data.get("timing", 0)
            + v
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Les poids doivent sommer à 1.0, got {total:.2f} "
                f"(fit={data.get('fit')}, activity={data.get('activity')}, "
                f"timing={data.get('timing')}, source={v})"
            )
        return v


class RevenueVelocityConfig(BaseAgentConfig):
    """Configuration de l'agent Revenue Velocity.

    Les seuils sont CALIBRÉS par l'orchestrateur selon :
    - Le cycle de vente moyen du client
    - Le volume de deals
    - Le secteur d'activité
    """

    agent_type: AgentType = AgentType.REVENUE_VELOCITY

    # ── Pipeline Truth ──
    stagnation_threshold_days: int = Field(
        default=21,
        ge=7,
        le=90,
        description=(
            "Jours sans activité pour considérer un deal comme stagnant. "
            "Calibré selon le cycle moyen : typiquement cycle_moyen × 0.6"
        ),
    )
    zombie_threshold_days: int = Field(
        default=45,
        ge=14,
        le=180,
        description="Jours sans activité pour tagger un deal comme zombie.",
    )
    pipeline_realistic_factor: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Facteur de correction du pipeline déclaré.",
    )

    # ── Lead Scoring ──
    score_weights: ScoreWeights = Field(
        default_factory=ScoreWeights,
    )
    hot_lead_threshold: int = Field(
        default=75, ge=50, le=95,
        description="Score minimum pour notification 'lead chaud'.",
    )
    archive_threshold: int = Field(
        default=20, ge=5, le=40,
        description="Score en dessous duquel recommander l'archivage.",
    )

    # ── Forecast ──
    forecast_correction_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Facteur de correction du forecast.",
    )

    # ── Alertes ──
    high_value_deal_threshold: float = Field(
        default=10000,
        ge=0,
        description="Montant € au-dessus duquel un deal est 'high value'.",
    )

    def to_agent_config(self) -> AgentConfig:
        """Convertit en AgentConfig avec thresholds + parameters."""
        return AgentConfig(
            **self._base_to_agent_config_kwargs(),
            data_sources=["hubspot", "gmail"],
            thresholds=[
                Threshold(
                    metric_name="stagnation_days",
                    warning_value=float(self.stagnation_threshold_days),
                    critical_value=float(self.zombie_threshold_days),
                    direction="above",
                ),
                Threshold(
                    metric_name="pipeline_realistic_value",
                    warning_value=self.pipeline_realistic_factor,
                    critical_value=self.pipeline_realistic_factor * 0.5,
                    direction="below",
                ),
                Threshold(
                    metric_name="lead_score",
                    warning_value=float(self.archive_threshold),
                    critical_value=float(self.archive_threshold * 0.5),
                    direction="below",
                ),
            ],
            parameters={
                "stagnation_threshold_days": self.stagnation_threshold_days,
                "zombie_threshold_days": self.zombie_threshold_days,
                "pipeline_realistic_factor": self.pipeline_realistic_factor,
                "score_weights": self.score_weights.model_dump(),
                "hot_lead_threshold": self.hot_lead_threshold,
                "archive_threshold": self.archive_threshold,
                "forecast_correction_factor": self.forecast_correction_factor,
                "high_value_deal_threshold": self.high_value_deal_threshold,
            },
        )


class ProcessClarityConfig(BaseAgentConfig):
    """Configuration de l'agent Process Clarity.

    Paramètres consommés par ProcessClarityAgent via config.parameters.
    """

    agent_type: AgentType = AgentType.PROCESS_CLARITY
    run_frequency: str = "weekly"

    # ── SOP Detection ──
    sop_min_occurrences: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Nombre minimum de répétitions pour générer une SOP.",
    )

    # ── Bottleneck ──
    bottleneck_time_multiplier: float = Field(
        default=2.0,
        ge=1.2,
        le=5.0,
        description=(
            "Un stage est un bottleneck si son temps > "
            "moyenne_des_autres_stages × ce multiplier."
        ),
    )
    concentration_threshold: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Risque de concentration si une personne > X% des process.",
    )

    # ── Deadlines ──
    deadline_warning_hours: int = Field(
        default=48,
        ge=12,
        le=168,
        description="Heures avant deadline pour envoyer un warning.",
    )
    deadline_escalation_days: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Jours de retard avant escalade automatique.",
    )

    # ── Check-ins ──
    checkin_time: str = Field(
        default="09:00",
        description="Heure d'envoi des check-ins quotidiens (HH:MM).",
    )

    # ── Doublons ──
    duplicate_detection_enabled: bool = True
    duplicate_auto_merge_threshold: float = Field(
        default=0.95,
        ge=0.80,
        le=1.0,
        description="Score au-dessus duquel on merge automatiquement.",
    )
    duplicate_confirm_threshold: float = Field(
        default=0.70,
        ge=0.50,
        le=0.95,
        description="Score au-dessus duquel on demande confirmation.",
    )
    ping_pong_threshold: int = Field(
        default=5,
        ge=3,
        le=15,
        description="Rebonds email/tâche pour détecter un ping-pong.",
    )

    # ── Charge ──
    max_tasks_per_person_warning: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Nombre max de tâches actives avant alerte surcharge.",
    )

    # ── Handoffs ──
    handoff_context_max_words: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Longueur max du résumé de contexte dans un handoff.",
    )

    # ── Coût ──
    hourly_rate_estimate: float = Field(
        default=50.0,
        ge=15,
        le=200,
        description="Taux horaire estimé pour chiffrer les gaspillages en €.",
    )

    def to_agent_config(self) -> AgentConfig:
        """Convertit en AgentConfig avec thresholds + parameters."""
        return AgentConfig(
            **self._base_to_agent_config_kwargs(),
            data_sources=["hubspot", "gmail", "project_management"],
            thresholds=[
                Threshold(
                    metric_name="cycle_time_days",
                    warning_value=7.0,
                    critical_value=14.0,
                    direction="above",
                ),
                Threshold(
                    metric_name="on_time_delivery_rate",
                    warning_value=0.80,
                    critical_value=0.60,
                    direction="below",
                ),
                Threshold(
                    metric_name="overdue_tasks_count",
                    warning_value=5.0,
                    critical_value=15.0,
                    direction="above",
                ),
                Threshold(
                    metric_name="duplicate_rate",
                    warning_value=0.05,
                    critical_value=0.15,
                    direction="above",
                ),
                Threshold(
                    metric_name="sop_occurrence_threshold",
                    warning_value=float(self.sop_min_occurrences),
                    critical_value=float(self.sop_min_occurrences),
                    direction="above",
                ),
                Threshold(
                    metric_name="workload_imbalance_ratio",
                    warning_value=2.0,
                    critical_value=3.0,
                    direction="above",
                ),
            ],
            parameters={
                "sop_min_occurrences": self.sop_min_occurrences,
                "bottleneck_time_multiplier": self.bottleneck_time_multiplier,
                "concentration_threshold": self.concentration_threshold,
                "deadline_warning_hours": self.deadline_warning_hours,
                "deadline_escalation_days": self.deadline_escalation_days,
                "checkin_time": self.checkin_time,
                "duplicate_detection_enabled": self.duplicate_detection_enabled,
                "duplicate_auto_merge_threshold": self.duplicate_auto_merge_threshold,
                "duplicate_confirm_threshold": self.duplicate_confirm_threshold,
                "ping_pong_threshold": self.ping_pong_threshold,
                "max_tasks_per_person_warning": self.max_tasks_per_person_warning,
                "handoff_context_max_words": self.handoff_context_max_words,
                "hourly_rate_estimate": self.hourly_rate_estimate,
            },
        )


class CashPredictabilityConfig(BaseAgentConfig):
    """Configuration de l'agent Cash Predictability."""

    agent_type: AgentType = AgentType.CASH_PREDICTABILITY
    run_frequency: str = "daily"

    # ── Seuils d'alerte ──
    cash_threshold_yellow_months: float = Field(
        default=3.0,
        ge=1.0,
        le=12.0,
        description="Alerte jaune si runway < X mois (scénario base).",
    )
    cash_threshold_red_months: float = Field(
        default=1.5,
        ge=0.5,
        le=6.0,
        description="Alerte rouge si runway < X mois (scénario stress).",
    )

    # ── Scénarios ──
    stress_pipeline_factor: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="En scénario stress, pipeline × ce facteur.",
    )
    stress_payment_delay_days: int = Field(
        default=15,
        ge=5,
        le=45,
        description="En scénario stress, retard de paiement additionnel.",
    )
    upside_pipeline_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="En scénario upside, pipeline × ce facteur.",
    )

    #nCorrection ──
    forecast_optimism_correction: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Facteur de correction si forecast biaisé.",
    )

    def to_agent_config(self) -> AgentConfig:
        """Convertit en AgentConfig avec thresholds + parameters."""
        return AgentConfig(
            **self._base_to_agent_config_kwargs(),
            data_sources=["quickbooks", "hubspot"],
            thresholds=[
                Threshold(
                    metric_name="runway_months",
                    warning_value=self.cash_threshold_yellow_months,
                    critical_value=self.cash_threshold_red_months,
                    direction="below",
                ),
                Threshold(
                    metric_name="forecast_accuracy",
                    warning_value=0.80,
                    critical_value=0.60,
                    direction="below",
                ),
            ],
            parameters={
                "cash_threshold_yellow_months": self.cash_threshold_yellow_months,
                "cash_threshold_red_months": self.cash_threshold_red_months,
                "stress_pipeline_factor": self.stress_pipeline_factor,
                "stress_payment_delay_days": self.stress_payment_delay_days,
                "upside_pipeline_factor": self.upside_pipeline_factor,
                "forecast_optimism_correction": self.forecast_optimism_correction,
            },
        )


class AcquisitionEfficiencyConfig(BaseAgentConfig):
    """Configuration de l'agent Acquisition Efficiency.

    Souvent DÉSACTIVÉ en V1 si le client n'a pas assez
    de données marketing trackées.
    """

    agent_type: AgentType = AgentType.ACQUISITION_EFFICIENCY
    run_frequency: str = "monthly"

    # ── Attribution ──
    attribution_model: str = Field(
        default="positional",
        description="'first_touch', 'last_touch', 'linear', 'positional'",
    )
    first_touch_weight: float = Field(default=0.3, ge=0, le=1.0)
    last_touch_weight: float = Field(default=0.3, ge=0, le=1.0)

    # ── Seuils ──
    cac_anomaly_threshold_pct: float = Field(
        default=0.25,
        ge=0.1,
        le=1.0,
        description="Alerte si le CAC d'un canal change de > X% en un mois.",
    )

    # ── Minimum de données ──
    min_clients_for_analysis: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Clients signés minimum pour analyse fiable.",
    )
    min_leads_per_channel: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Leads minimum par canal pour scorer le canal.",
    )

    @field_validator("attribution_model")
    @classmethod
    def validate_attribution(cls, v: str) -> str:
        allowed = {"first_touch", "last_touch", "linear", "positional"}
        if v not in allowed:
            raise ValueError(f"attribution_model must be one of {allowed}")
        return v

    def to_agent_config(self) -> AgentConfig:
        """Convertit en AgentConfig avec thresholds + parameters."""
        return AgentConfig(
            **self._base_to_agent_config_kwargs(),
            data_sources=["hubspot", "gmail"],
            thresholds=[
                Threshold(
                    metric_name="blended_cac",
                    warning_value=500.0,
                    critical_value=1000.0,
                    direction="above",
                ),
                Threshold(
                    metric_name="cac_change_pct",
                    warning_value=self.cac_anomaly_threshold_pct,
                    critical_value=self.cac_anomaly_threshold_pct * 2,
                    direction="above",
                ),
            ],
            parameters={
                "attribution_model": self.attribution_model,
                "first_touch_weight": self.first_touch_weight,
                "last_touch_weight": self.last_touch_weight,
                "cac_anomaly_threshold_pct": self.cac_anomaly_threshold_pct,
                "min_clients_for_analysis": self.min_clients_for_analysis,
                "min_leads_per_channel": self.min_leads_per_channel,
            },
        )


# ══════════════════════════════════════════════════════════════
# SET COMPLET
# ══════════════════════════════════════════════════════════════


class AgentConfigSet(BaseModel):
    """
    Ensemble complet des configurations d'agents pour un client.

    Stocké dans la table agent_configs, sérialisé en JSON.
    L'orchestrateur crée ce set après le scan.
    L'Adaptateur le recalibre chaque semaine.
    """

    company_id: str = Field(..., min_length=1)
    revenue_velocity: RevenueVelocityConfig
    process_clarity: ProcessClarityConfig
    cash_predictability: CashPredictabilityConfig
    acquisition_efficiency: AcquisitionEfficiencyConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def enabled_agents(self) -> list[AgentType]:
        """Liste des agents activés pour ce client."""
        agents = []
        for config in [
            self.revenue_velocity,
            self.process_clarity,
            self.cash_predictability,
            self.acquisition_efficiency,
        ]:
            if config.enabled:
                agents.append(config.agent_type)
        return agents

         @property
    def enabled_count(self) -> int:
        return len(self.enabled_agents)

    def get_config(self, agent_type: AgentType) -> BaseAgentConfig | None:
        """Récupère la config typée forte d'un agent par son type."""
        mapping: dict[AgentType, BaseAgentConfig] = {
            AgentType.REVENUE_VELOCITY: self.revenue_velocity,
            AgentType.PROCESS_CLARITY: self.process_clarity,
            AgentType.CASH_PREDICTABILITY: self.cash_predictability,
            AgentType.ACQUISITION_EFFICIENCY: self.acquisition_efficiency,
        }
        return mapping.get(agent_type)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig | None:
        """
        Récupère la config au format générique AgentConfig.

        C'est ce format que les agents consomment dans leur __init__.
        """
        typed_config = self.get_config(agent_type)
        if typed_config is None:
            return None
        return typed_config.to_agent_config()

    def to_agent_configs(self) -> dict[AgentType, AgentConfig]:
        """Génère tous les AgentConfig pour les agents activés."""
        return {
            agent_type: self.get_agent_config(agent_type)
            for agent_type in self.enabled_agents
            if self.get_agent_config(agent_type) is not None
        }











