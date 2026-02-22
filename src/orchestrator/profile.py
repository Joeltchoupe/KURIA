"""
ProfileGenerator — Scan → CompanyProfile → AgentConfigSet.

Après le scan initial, le ProfileGenerator :
  1. Analyse les données collectées
  2. Génère le CompanyProfile
  3. Calcule le Score de Clarté
  4. Configure les 4 agents avec des seuils calibrés

C'est le JOUR 0 du client.
Le ProfileGenerator ne tourne qu'une fois (+ reset possible).

Design decisions :
  - Les configs sont calibrées selon le secteur, la taille, les données
  - Un agent peut être désactivé si les données sont insuffisantes
  - Le Score de Clarté initial est la baseline pour mesurer la progression
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from models.agent_config import (
    AgentConfigSet,
    AgentType,
    RevenueVelocityConfig,
    ProcessClarityConfig,
    CashPredictabilityConfig,
    AcquisitionEfficiencyConfig,
)
from models.company import CompanyProfile, CompanySize


# ══════════════════════════════════════════════════════════════
# SCAN RESULT (input du ProfileGenerator)
# ══════════════════════════════════════════════════════════════


class ScanResult(BaseModel):
    """
    Résultat brut du scan initial.

    Contient les métriques brutes extraites des connecteurs.
    Le ProfileGenerator les transforme en profil + configs.
    """
    company_id: str
    company_name: str

    # Données CRM
    has_crm: bool = False
    crm_provider: str | None = None
    total_deals: int = Field(default=0, ge=0)
    avg_deal_cycle_days: float | None = None
    total_contacts: int = Field(default=0, ge=0)
    pipeline_value: float = Field(default=0, ge=0)

    # Données email
    has_email: bool = False
    email_volume_30d: int = Field(default=0, ge=0)

    # Données finance
    has_finance: bool = False
    finance_provider: str | None = None
    monthly_revenue: float | None = None
    monthly_expenses: float | None = None
    cash_balance: float | None = None
    outstanding_invoices: float | None = None
    avg_payment_delay_days: float | None = None

    # Données marketing
    has_marketing_data: bool = False
    marketing_channels: list[str] = Field(default_factory=list)
    total_leads_30d: int = Field(default=0, ge=0)
    marketing_spend_30d: float | None = None

    # Méta
    scanned_at: datetime = Field(default_factory=datetime.utcnow)
    scan_duration_seconds: float = Field(default=0.0, ge=0)
    data_quality_score: float = Field(default=0.5, ge=0, le=1)

    @property
    def estimated_company_size(self) -> CompanySize:
        """Estime la taille de l'entreprise depuis les données."""
        if self.total_contacts > 500 or (self.monthly_revenue and self.monthly_revenue > 500000):
            return CompanySize.MEDIUM
        if self.total_contacts > 50 or (self.monthly_revenue and self.monthly_revenue > 50000):
            return CompanySize.SMALL
        return CompanySize.MICRO

    @property
    def data_completeness(self) -> float:
        """Score de complétude des données (0-1)."""
        checks = [
            self.has_crm,
            self.has_email,
            self.has_finance,
            self.has_marketing_data,
            self.total_deals > 0,
            self.avg_deal_cycle_days is not None,
            self.monthly_revenue is not None,
            self.cash_balance is not None,
        ]
        return sum(checks) / len(checks)


# ══════════════════════════════════════════════════════════════
# PROFILE GENERATOR
# ══════════════════════════════════════════════════════════════


class ProfileGenerator:
    """
    Génère le profil client + configs agents depuis le scan.

    Usage :
        generator = ProfileGenerator()
        config_set = generator.generate(scan_result)
    """

    def __init__(self) -> None:
        self._generation_history: list[dict[str, Any]] = []

    def generate(self, scan: ScanResult) -> AgentConfigSet:
        """
        Génère l'AgentConfigSet complet depuis un ScanResult.

        Calibre chaque agent selon les données disponibles.
        Désactive les agents sans données suffisantes.
        """
        config_set = AgentConfigSet(
            company_id=scan.company_id,
            revenue_velocity=self._configure_revenue_velocity(scan),
            process_clarity=self._configure_process_clarity(scan),
            cash_predictability=self._configure_cash_predictability(scan),
            acquisition_efficiency=self._configure_acquisition_efficiency(scan),
        )

        self._generation_history.append({
            "company_id": scan.company_id,
            "generated_at": datetime.utcnow().isoformat(),
            "enabled_agents": [a.value for a in config_set.enabled_agents],
            "data_completeness": scan.data_completeness,
        })

        return config_set

    # ──────────────────────────────────────────────────────
    # CALIBRATION PAR AGENT
    # ──────────────────────────────────────────────────────

    def _configure_revenue_velocity(
        self, scan: ScanResult
    ) -> RevenueVelocityConfig:
        """Calibre Revenue Velocity selon les données CRM."""
        enabled = scan.has_crm and scan.total_deals >= 5

        # Calibrer stagnation_threshold selon le cycle moyen
        if scan.avg_deal_cycle_days and scan.avg_deal_cycle_days > 0:
            stagnation = max(7, min(90, int(scan.avg_deal_cycle_days * 0.6)))
            zombie = max(14, min(180, int(scan.avg_deal_cycle_days * 1.2)))
        else:
            stagnation = 21
            zombie = 45

        # Calibrer high_value_deal selon le pipeline
        if scan.pipeline_value > 0 and scan.total_deals > 0:
            avg_deal = scan.pipeline_value / scan.total_deals
            high_value = max(1000, avg_deal * 2)
        else:
            high_value = 10000

        return RevenueVelocityConfig(
            company_id=scan.company_id,
            enabled=enabled,
            stagnation_threshold_days=stagnation,
            zombie_threshold_days=zombie,
            high_value_deal_threshold=high_value,
        )

    def _configure_process_clarity(
        self, scan: ScanResult
    ) -> ProcessClarityConfig:
        """Calibre Process Clarity selon les données ops."""
        # Process Clarity est presque toujours activé
        # Il peut fonctionner avec juste le CRM + email
        enabled = scan.has_crm or scan.has_email

        # Ajuster hourly_rate selon la taille
        size = scan.estimated_company_size
        hourly_rates = {
            CompanySize.MICRO: 35.0,
            CompanySize.SMALL: 50.0,
            CompanySize.MEDIUM: 65.0,
        }
        hourly_rate = hourly_rates.get(size, 50.0)

        # Plus de tâches = seuil de surcharge plus haut
        max_tasks = 10
        if scan.total_deals > 100:
            max_tasks = 15
        elif scan.total_deals > 50:
            max_tasks = 12

        return ProcessClarityConfig(
            company_id=scan.company_id,
            enabled=enabled,
            hourly_rate_estimate=hourly_rate,
            max_tasks_per_person_warning=max_tasks,
        )

    def _configure_cash_predictability(
        self, scan: ScanResult
    ) -> CashPredictabilityConfig:
        """Calibre Cash Predictability selon les données finance."""
        enabled = scan.has_finance and scan.monthly_revenue is not None

        # Ajuster stress_payment_delay selon les données réelles
        stress_delay = 15
        if scan.avg_payment_delay_days is not None:
            stress_delay = max(5, min(45, int(scan.avg_payment_delay_days * 1.5)))

        return CashPredictabilityConfig(
            company_id=scan.company_id,
            enabled=enabled,
            stress_payment_delay_days=stress_delay,
        )

    def _configure_acquisition_efficiency(
        self, scan: ScanResult
    ) -> AcquisitionEfficiencyConfig:
        """Calibre Acquisition Efficiency selon les données marketing."""
        # Souvent désactivé en V1
        enabled = (
            scan.has_marketing_data
            and scan.total_leads_30d >= 10
            and len(scan.marketing_channels) >= 2
        )

        return AcquisitionEfficiencyConfig(
            company_id=scan.company_id,
            enabled=enabled,
        )

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    @property
    def generation_count(self) -> int:
        return len(self._generation_history)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._generation_history)
