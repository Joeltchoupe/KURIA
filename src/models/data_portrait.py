"""
DataPortrait — Le résultat complet du scan Phase 1.

C'est le BRIEF DU CONSULTANT pour la session MIRROR.
Jamais montré tel quel au client.
Contient tout ce que l'agent Scanner a trouvé.

Design decisions :
- Chaque section (sales, ops, finance, marketing) est un modèle séparé
- Les anomalies sont typées et structurées (pas du texte libre)
- Les coûts estimés sont toujours accompagnés d'un niveau de confiance
- Le portrait est versionné (le même client peut être re-scanné)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from models.clarity_score import ClarityScore


# ──────────────────────────────────────────────
# ANOMALIES
# ──────────────────────────────────────────────


class AnomalyType(str, Enum):
    """Types d'anomalies détectables par le Scanner."""

    CONCENTRATION = "concentration"        # Risque de dépendance à 1-2 personnes
    STAGNATION = "stagnation"              # Données/deals qui ne bougent plus
    INCONSISTENCY = "inconsistency"        # Déclaré ≠ réel
    GAP = "gap"                            # Données manquantes
    OUTLIER = "outlier"                    # Valeur anormalement haute/basse
    PATTERN = "pattern"                    # Pattern récurrent problématique


class Anomaly(BaseModel):
    """Une anomalie détectée pendant le scan."""

    type: AnomalyType
    department: str = Field(
        ...,
        description="Département concerné : 'sales', 'ops', 'finance', 'marketing'",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Titre court et factuel",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Description détaillée avec les chiffres",
    )
    evidence: str = Field(
        ...,
        min_length=1,
        description="Les données brutes qui prouvent l'anomalie",
    )
    severity: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Sévérité 0-1. > 0.7 = critique.",
    )
    estimated_annual_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Coût annuel estimé en euros (si chiffrable)",
    )
    cost_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Confiance dans l'estimation du coût",
    )

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        allowed = {"sales", "ops", "finance", "marketing", "general"}
        if v not in allowed:
            raise ValueError(f"department must be one of {allowed}, got '{v}'")
        return v


# ──────────────────────────────────────────────
# ANALYSES PAR DÉPARTEMENT
# ──────────────────────────────────────────────


class PersonDependency(BaseModel):
    """Une personne identifiée comme point de concentration/risque."""

    name: str
    role: Optional[str] = None
    involvement_pct: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Pourcentage des process/deals où cette personne intervient",
    )


class SalesAnalysis(BaseModel):
    """Analyse complète des données commerciales."""

    # ── Pipeline ──
    pipeline_total_declared: float = Field(..., ge=0)
    pipeline_total_realistic: float = Field(..., ge=0)
    pipeline_gap_ratio: float = Field(
        ...,
        ge=0,
        description="declared / realistic. > 2 = problème sérieux.",
    )

    # ── Deals ──
    deals_active: int = Field(..., ge=0)
    deals_stagnant: int = Field(..., ge=0)
    deals_without_next_step: int = Field(..., ge=0)
    stagnation_threshold_days: int = Field(
        ...,
        gt=0,
        description="Seuil utilisé pour détecter la stagnation",
    )

    # ── Cycle ──
    avg_cycle_days: float = Field(..., ge=0)
    bottleneck_stage: Optional[str] = None
    bottleneck_stage_avg_days: Optional[float] = Field(default=None, ge=0)

    # ── Performance ──
    win_rate_90d: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Taux de conversion sur les 90 derniers jours",
    )
    forecast_confidence: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Confiance dans le forecast CRM actuel",
    )

    # ── Concentration ──
    concentration_risk: str = Field(
        ...,
        description="'low', 'medium', 'high'",
    )
    key_people: list[PersonDependency] = Field(default_factory=list)

    # ── Anomalies ──
    anomalies: list[Anomaly] = Field(default_factory=list)

    @field_validator("concentration_risk")
    @classmethod
    def validate_concentration_risk(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"concentration_risk must be one of {allowed}")
        return v

    @property
    def stagnation_rate(self) -> float:
        if self.deals_active == 0:
            return 0.0
        return self.deals_stagnant / self.deals_active


class OperationsAnalysis(BaseModel):
    """Analyse complète des opérations."""

    # ── Communication ──
    email_response_internal_avg_hours: Optional[float] = Field(default=None, ge=0)
    email_response_external_avg_hours: Optional[float] = Field(default=None, ge=0)
    after_hours_activity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Pourcentage d'activité en dehors des heures de bureau",
    )

    # ── Tâches ──
    tasks_overdue: Optional[int] = Field(default=None, ge=0)
    tasks_unassigned: Optional[int] = Field(default=None, ge=0)

    # ── Process ──
    process_documentation_score: str = Field(
        default="unknown",
        description="'high', 'medium', 'low', 'unknown'",
    )

    # ── Dépendances ──
    key_person_dependencies: list[PersonDependency] = Field(default_factory=list)

    # ── Anomalies ──
    anomalies: list[Anomaly] = Field(default_factory=list)

    @field_validator("process_documentation_score")
    @classmethod
    def validate_doc_score(cls, v: str) -> str:
        allowed = {"high", "medium", "low", "unknown"}
        if v not in allowed:
            raise ValueError(f"process_documentation_score must be one of {allowed}")
        return v


class TopExpense(BaseModel):
    """Un poste de dépense majeur."""

    category: str
    monthly_amount: float = Field(..., ge=0)
    annual_amount: float = Field(..., ge=0)
    pct_of_total: float = Field(..., ge=0, le=1.0)


class FinanceAnalysis(BaseModel):
    """Analyse complète des données financières."""

    # ── Cash ──
    cash_position: float = Field(
        ...,
        description="Position cash actuelle en euros (peut être négatif)",
    )
    monthly_burn: float = Field(..., ge=0, description="Burn rate mensuel en euros")
    runway_months: float = Field(
        ...,
        ge=0,
        description="Nombre de mois de runway au rythme actuel",
    )

    # ── Paiements ──
    avg_client_payment_days: float = Field(..., ge=0)
    avg_supplier_payment_days: float = Field(..., ge=0)
    cash_gap_days: float = Field(
        ...,
        description="Écart en jours entre paiements clients et fournisseurs. Positif = tension.",
    )

    # ── Factures ──
    invoices_overdue_count: int = Field(..., ge=0)
    invoices_overdue_value: float = Field(..., ge=0)

    # ── Structure revenu ──
    revenue_recurring_pct: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Part du revenu récurrent dans le CA total",
    )

    # ── Dépenses ──
    top_expenses: list[TopExpense] = Field(default_factory=list)

    # ── Anomalies ──
    anomalies: list[Anomaly] = Field(default_factory=list)

    @property
    def is_cash_critical(self) -> bool:
        """Runway < 3 mois = critique."""
        return self.runway_months < 3.0

    @property
    def cash_gap_cost_annual(self) -> float:
        """Estimation du coût annuel du gap de trésorerie.
        Formule simplifiée : gap_days × burn_daily × taux d'emprunt implicite."""
        if self.cash_gap_days <= 0:
            return 0.0
        daily_burn = self.monthly_burn / 30
        # Coût d'opportunité estimé à 8% annuel
        return self.cash_gap_days * daily_burn * 0.08


class MarketingAnalysis(BaseModel):
    """Analyse des données marketing/acquisition.
    Souvent la plus INCOMPLÈTE car les PME trackent mal l'acquisition."""

    # ── Sources ──
    has_source_tracking: bool = Field(
        default=False,
        description="Le client track-t-il la source de ses leads ?",
    )
    identified_channels: list[str] = Field(default_factory=list)
    leads_with_source_pct: float = Field(
        default=0.0,
        ge=0,
        le=1.0,
        description="Pourcentage de leads avec une source identifiée",
    )

    # ── CAC ──
    blended_cac: Optional[float] = Field(
        default=None,
        ge=0,
        description="CAC blended si calculable",
    )
    cac_by_channel: Optional[dict[str, float]] = None

    # ── Volume ──
    monthly_leads_avg: Optional[int] = Field(default=None, ge=0)
    lead_to_client_conversion: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
    )

    # ── Anomalies ──
    anomalies: list[Anomaly] = Field(default_factory=list)

    @property
    def data_quality(self) -> str:
        """Qualité des données marketing disponibles."""
        if not self.has_source_tracking or self.leads_with_source_pct < 0.3:
            return "insufficient"
        if self.leads_with_source_pct < 0.7:
            return "partial"
        return "good"


# ──────────────────────────────────────────────
# DATA PORTRAIT COMPLET
# ──────────────────────────────────────────────


class DataPortrait(BaseModel):
    """Le Data Portrait complet — output de la Phase 1 du scan.

    Ce document est :
    1. Le brief du consultant pour la session MIRROR
    2. L'input de l'orchestrateur pour configurer les agents
    3. La baseline contre laquelle mesurer les progrès

    Il n'est JAMAIS montré tel quel au client.
    """

    # ── Identité ──
    company_id: str = Field(..., min_length=1)
    scan_date: datetime = Field(default_factory=datetime.utcnow)
    scan_version: int = Field(default=1, ge=1)
    tools_connected: list[str] = Field(
        ...,
        min_length=1,
        description="Au moins un outil doit être connecté",
    )

    # ── Analyses ──
    sales: SalesAnalysis
    operations: OperationsAnalysis
    finance: FinanceAnalysis
    marketing: MarketingAnalysis

    # ── Score ──
    clarity_score: ClarityScore

    # ── Coût total estimé ──
    estimated_total_implicit_cost: float = Field(
        ...,
        ge=0,
        description="Somme des coûts annuels de toutes les frictions identifiées",
    )

    # ── Anomalies agrégées ──

    @property
    def all_anomalies(self) -> list[Anomaly]:
        """Toutes les anomalies de tous les départements, triées par sévérité."""
        anomalies = (
            self.sales.anomalies
            + self.operations.anomalies
            + self.finance.anomalies
            + self.marketing.anomalies
        )
        return sorted(anomalies, key=lambda a: a.severity, reverse=True)

    @property
    def critical_anomalies(self) -> list[Anomaly]:
        """Anomalies avec sévérité > 0.7."""
        return [a for a in self.all_anomalies if a.severity > 0.7]

    @property
    def total_chiffrable_cost(self) -> float:
        """Somme des coûts des anomalies qui ont un coût estimé."""
        return sum(
            a.estimated_annual_cost
            for a in self.all_anomalies
            if a.estimated_annual_cost is not None
        )

    def summary_for_consultant(self) -> dict:
        """Résumé structuré pour préparer la session MIRROR."""
        return {
            "company_id": self.company_id,
            "clarity_score": self.clarity_score.overall,
            "grade": self.clarity_score.grade,
            "weakest_dept": self.clarity_score.by_department.weakest_department,
            "pipeline_gap": self.sales.pipeline_gap_ratio,
            "cash_runway_months": self.finance.runway_months,
            "critical_anomalies_count": len(self.critical_anomalies),
            "estimated_total_cost": self.estimated_total_implicit_cost,
  }
