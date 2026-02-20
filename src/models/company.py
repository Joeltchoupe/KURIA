"""
CompanyProfile — Le profil complet d'un client Kuria.

C'est le document CENTRAL que l'orchestrateur lit
pour configurer chaque agent. Généré par le scan,
enrichi en continu par les agents.

Design decisions :
- Pydantic v2 pour validation stricte + sérialisation JSON native
- Enums pour tout ce qui est catégoriel (pas de strings libres)
- Optional explicite pour tout ce qui peut manquer en V1
- Validators custom pour les règles métier
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ──────────────────────────────────────────────
# ENUMS
# ──────────────────────────────────────────────


class GrowthStage(str, Enum):
    """Stade de maturité de l'entreprise.
    Détermine les benchmarks et seuils des agents."""

    STARTUP = "startup"          # < 10 personnes, produit pas stabilisé
    GROWING = "growing"          # 10-50, traction, process informels
    SCALING = "scaling"          # 50-200, structuration en cours
    MATURE = "mature"            # 200+, process établis, optimisation


class CompanySize(str, Enum):
    """Tranche d'effectif. Utilisé pour les benchmarks sectoriels."""

    XS = "1-10"
    S = "11-25"
    M = "26-50"
    L = "51-100"
    XL = "101-250"
    XXL = "251-500"


class ToolProvider(str, Enum):
    """Outils supportés en V1.
    Chaque ajout ici nécessite un connecteur dans connectors/."""

    HUBSPOT = "hubspot"
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    QUICKBOOKS = "quickbooks"
    PENNYLANE = "pennylane"
    NOTION = "notion"
    SLACK = "slack"
    ASANA = "asana"


# ──────────────────────────────────────────────
# TOOL CONNECTION
# ──────────────────────────────────────────────


class ToolConnection(BaseModel):
    """État de connexion d'un outil client.
    Stocke le statut OAuth, pas les tokens (ceux-ci sont en DB chiffrés)."""

    provider: ToolProvider
    connected: bool = False
    plan: Optional[str] = None          # "free", "pro", "enterprise"
    connected_at: Optional[datetime] = None
    last_sync_at: Optional[datetime] = None
    last_sync_status: Optional[str] = None  # "success", "partial", "failed"
    record_count: Optional[int] = None  # Nombre de records au dernier sync

    @field_validator("last_sync_status")
    @classmethod
    def validate_sync_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("success", "partial", "failed"):
            raise ValueError(
                f"last_sync_status must be 'success', 'partial', or 'failed', got '{v}'"
            )
        return v


class ConnectedTools(BaseModel):
    """Ensemble des outils connectés pour un client.
    L'orchestrateur lit ça pour savoir quels agents activer."""

    crm: Optional[ToolConnection] = None
    email: Optional[ToolConnection] = None
    finance: Optional[ToolConnection] = None
    project: Optional[ToolConnection] = None
    chat: Optional[ToolConnection] = None

    @property
    def connected_list(self) -> list[ToolProvider]:
        """Retourne la liste des providers effectivement connectés."""
        tools = []
        for field_name in ["crm", "email", "finance", "project", "chat"]:
            conn = getattr(self, field_name)
            if conn is not None and conn.connected:
                tools.append(conn.provider)
        return tools

    @property
    def connected_count(self) -> int:
        return len(self.connected_list)

    def has_minimum_for_scan(self) -> bool:
        """Le scan requiert au minimum un CRM connecté."""
        return self.crm is not None and self.crm.connected


# ──────────────────────────────────────────────
# COMPANY PROFILE
# ──────────────────────────────────────────────


class CompanyProfile(BaseModel):
    """Profil complet d'un client Kuria.

    Créé lors du scan initial (Phase 1).
    Enrichi par la session consulting (Phase 2).
    Mis à jour mensuellement par l'Adaptateur.

    Ce modèle est la SOURCE DE VÉRITÉ pour :
    - L'orchestrateur (configuration des agents)
    - Le dashboard (affichage)
    - Le rapport hebdo (contexte)
    """

    # ── Identité ──
    company_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifiant unique (UUID ou slug)",
    )
    name: str = Field(..., min_length=1, max_length=256)
    sector: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Secteur d'activité (ex: 'b2b_services', 'ecommerce', 'saas')",
    )
    sub_sector: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Sous-secteur (ex: 'consulting', 'logistics')",
    )
    size: CompanySize
    employee_count: int = Field(..., gt=0, le=10000)
    annual_revenue: float = Field(
        ...,
        ge=0,
        description="CA annuel en euros",
    )
    growth_stage: GrowthStage

    # ── Outils ──
    tools: ConnectedTools

    # ── Scores (remplis après le scan) ──
    clarity_score: Optional["ClarityScore"] = None

    # ── Méta ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    scan_completed_at: Optional[datetime] = None
    consulting_completed_at: Optional[datetime] = None
    agents_deployed_at: Optional[datetime] = None
    version: int = Field(
        default=1,
        ge=1,
        description="Version du profil. Incrémentée à chaque recalibration.",
    )

    # ── Validators ──

    @field_validator("annual_revenue")
    @classmethod
    def validate_revenue_range(cls, v: float) -> float:
        """Le produit cible les PME 2-100M€.
        On accepte en dehors mais on log un warning."""
        # Pas de rejet — on accepte tout, le scan décidera
        return v

    @model_validator(mode="after")
    def validate_coherence(self) -> "CompanyProfile":
        """Vérifie la cohérence taille / CA / stade."""
        if self.growth_stage == GrowthStage.STARTUP and self.employee_count > 50:
            raise ValueError(
                f"Incohérence : growth_stage='startup' mais {self.employee_count} employés"
            )
        if self.growth_stage == GrowthStage.MATURE and self.employee_count < 50:
            raise ValueError(
                f"Incohérence : growth_stage='mature' mais seulement {self.employee_count} employés"
            )
        return self

    # ── Méthodes ──

    def is_scan_complete(self) -> bool:
        return self.scan_completed_at is not None

    def is_fully_onboarded(self) -> bool:
        return all([
            self.scan_completed_at,
            self.consulting_completed_at,
            self.agents_deployed_at,
        ])

    def days_since_deployment(self) -> Optional[int]:
        if self.agents_deployed_at is None:
            return None
        delta = datetime.utcnow() - self.agents_deployed_at
        return delta.days

    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "acme-corp-001",
                "name": "Acme Corp",
                "sector": "b2b_services",
                "sub_sector": "consulting",
                "size": "26-50",
                "employee_count": 35,
                "annual_revenue": 4_200_000,
                "growth_stage": "scaling",
                "tools": {
                    "crm": {
                        "provider": "hubspot",
                        "connected": True,
                        "plan": "pro",
                    },
                    "email": {
                        "provider": "gmail",
                        "connected": True,
                    },
                    "finance": {
                        "provider": "quickbooks",
                        "connected": True,
                    },
                },
                "version": 1,
            }
        }
