"""
AgentConfig — Configuration de chaque agent par client.

Généré par l'orchestrateur (ClientProfileGenerator) après le scan.
Recalibré chaque semaine par l'Adaptateur.

Chaque config contient les SEUILS et PARAMÈTRES spécifiques
au contexte de l'entreprise. C'est ce qui fait que le même agent
se comporte différemment pour chaque client.

C'est LE MOAT : l'orchestration contextuelle.

Design decisions :
- Un modèle de config par type d'agent (pas un dict générique)
- Chaque paramètre a une valeur par défaut raisonnable
- Chaque paramètre a des bornes (min/max) pour éviter les dérives
- L'Adaptateur ne peut ajuster que dans ces bornes
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AgentType(str, Enum):
    """Les 4 agents + le scanner."""

    SCANNER = "scanner"
    REVENUE_VELOCITY = "revenue_velocity"
    PROCESS_CLARITY = "process_clarity"
    CASH_PREDICTABILITY = "cash_predictability"
    ACQUISITION_EFFICIENCY = "acquisition_efficiency"


class BaseAgentConfig(BaseModel):
    """Configuration commune à tous les agents."""

    agent_type: AgentType
    enabled: bool = True
    company_id: str = Field(..., min_length=1)

    # ── LLM ──
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Modèle Claude à utiliser. Sonnet pour l'analyse, Haiku pour le routing.",
    )
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    temperature: float = Field(default=0.3, ge=0, le=1.0)

    # ── Fréquence ──
    run_frequency: str = Field(
        default="daily",
        description="'realtime', 'hourly', 'daily', 'weekly', 'monthly'",
    )

    # ── Notifications ──
    alert_channel: str = Field(
        default="email",
        description="'email', 'slack', 'both'",
    )
    alert_recipients: list[str] = Field(
        default_factory=list,
        description="Emails ou Slack IDs des destinataires",
    )

    # ── Confiance ──
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description=(
            "Seuil de confiance minimum pour que l'agent publie ses résultats. "
            "En dessous, il signale un manque de données."
        ),
    )

    # ── Méta ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    adaptation_count: int = Field(
        default=0,
        ge=0,
        description="Nombre de fois que la config a été recalibrée",
    )

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


# ──────────────────────────────────────────────
# CONFIGS SPÉCIFIQUES
# ──────────────────────────────────────────────


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
            "Nombre de jours sans activité pour considérer un deal comme stagnant. "
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
        description="Facteur de correction du pipeline déclaré. Ajusté par l'Adaptateur.",
    )

    # ── Lead Scoring ──
    score_weights: "ScoreWeights" = Field(
        default_factory=lambda: ScoreWeights(),
    )
    hot_lead_threshold: int = Field(
        default=75,
        ge=50,
        le=95,
        description="Score minimum pour déclencher une notification 'lead chaud'.",
    )
    archive_threshold: int = Field(
        default=20,
        ge=5,
        le=40,
        description="Score en dessous duquel recommander l'archivage.",
    )

    # ── Forecast ──
    forecast_correction_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description=(
            "Facteur de correction du forecast. "
            "Si le forecast est systématiquement optimiste de 15%, ce facteur = 0.85. "
            "Ajusté automatiquement par l'Adaptateur."
        ),
    )

    # ── Alertes ──
    high_value_deal_threshold: float = Field(
        default=10000,
        ge=0,
        description="Montant en € au-dessus duquel un deal est considéré 'high value'.",
    )


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
        total = data.get("fit", 0) + data.get("activity", 0) + data.get("timing", 0) + v
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Les poids doivent sommer à 1.0, got {total:.2f} "
                f"(fit={data.get('fit')}, activity={data.get('activity')}, "
                f"timing={data.get('timing')}, source={v})"
            )
        return v


class ProcessClarityConfig(BaseAgentConfig):
    """Configuration de l'agent Process Clarity."""

    agent_type: AgentType = AgentType.PROCESS_CLARITY

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
        description=(
            "Si une personne est impliquée dans > X% des process, "
            "c'est un risque de concentration."
        ),
    )

    # ── Waste ──
    duplicate_detection_enabled: bool = True
    ping_pong_threshold: int = Field(
        default=5,
        ge=3,
        le=15,
        description="Nombre de rebonds email/tâche pour détecter un ping-pong.",
    )

    # ── Coût ──
    hourly_rate_estimate: float = Field(
        default=50.0,
        ge=15,
        le=200,
        description=(
            "Taux horaire estimé pour chiffrer les gaspillages en €. "
            "Calibré selon la taille et le secteur."
        ),
    )

    run_frequency: str = "weekly"


class CashPredictabilityConfig(BaseAgentConfig):
    """Configuration de l'agent Cash Predictability."""

    agent_type: AgentType = AgentType.CASH_PREDICTABILITY

    # ── Seuils d'alerte ──
    cash_threshold_yellow_months: float = Field(
        default=3.0,
        ge=1.0,
        le=12.0,
        description="Alerte jaune si runway < X mois dans le scénario base.",
    )
    cash_threshold_red_months: float = Field(
        default=1.5,
        ge=0.5,
        le=6.0,
        description="Alerte rouge si runway < X mois dans le scénario stress.",
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

    # ── Correction ──
    forecast_optimism_correction: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description=(
            "Facteur de correction si le forecast est systématiquement biaisé. "
            "< 1.0 = on était trop optimiste. Ajusté par l'Adaptateur."
        ),
    )

    run_frequency: str = "daily"


class AcquisitionEfficiencyConfig(BaseAgentConfig):
    """Configuration de l'agent Acquisition Efficiency.

    Souvent DÉSACTIVÉ en V1 si le client n'a pas assez
    de données marketing trackées.
    """

    agent_type: AgentType = AgentType.ACQUISITION_EFFICIENCY

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
        description="Nombre minimum de clients signés pour que l'analyse soit fiable.",
    )
    min_leads_per_channel: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Nombre minimum de leads par canal pour scorer le canal.",
    )

    run_frequency: str = "monthly"

    @field_validator("attribution_model")
    @classmethod
    def validate_attribution(cls, v: str) -> str:
        allowed = {"first_touch", "last_touch", "linear", "positional"}
        if v not in allowed:
            raise ValueError(f"attribution_model must be one of {allowed}")
        return v


# ──────────────────────────────────────────────
# SET COMPLET
# ──────────────────────────────────────────────


class AgentConfigSet(BaseModel):
    """Ensemble complet des configurations d'agents pour un client.
    Stocké dans la table agent_configs, sérialisé en JSON."""

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

    def get_config(self, agent_type: AgentType) -> Optional[BaseAgentConfig]:
        """Récupère la config d'un agent par son type."""
        mapping = {
            AgentType.REVENUE_VELOCITY: self.revenue_velocity,
            AgentType.PROCESS_CLARITY: self.process_clarity,
            AgentType.CASH_PREDICTABILITY: self.cash_predictability,
            AgentType.ACQUISITION_EFFICIENCY: self.acquisition_efficiency,
        }
        return mapping.get(agent_type)
