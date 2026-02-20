"""
Friction & FrictionMap — Les frictions identifiées chez le client.

Une friction = une perte mesurable causée par un process
implicite, un trou de données, ou une inefficience structurelle.

La FrictionMap est le livrable central de la session MIRROR.
Elle est construite en live avec le CEO, puis affinée
par l'orchestrateur.

Design decisions :
- Chaque friction a un coût estimé ET un niveau de confiance
- La priorisation utilise la matrice impact × difficulté (inspiration BCG)
- Chaque friction est liée à un agent recommandé
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FrictionSeverity(str, Enum):
    """Sévérité de la friction basée sur le coût annuel estimé."""

    LOW = "low"             # < 25K€/an
    MEDIUM = "medium"       # 25-100K€/an
    HIGH = "high"           # 100-250K€/an
    CRITICAL = "critical"   # > 250K€/an


class PriorityQuadrant(str, Enum):
    """Position dans la matrice impact × difficulté.
    Détermine l'ordre d'attaque."""

    QUICK_WIN = "quick_win"             # Impact élevé + Facile → FAIRE EN PREMIER
    STRATEGIC = "strategic"             # Impact élevé + Difficile → PLANIFIER
    NICE_TO_HAVE = "nice_to_have"       # Impact faible + Facile → SI ON A LE TEMPS
    IGNORE = "ignore"                   # Impact faible + Difficile → PAS MAINTENANT


class Friction(BaseModel):
    """Une friction identifiée dans l'entreprise.

    Créée par le ScannerAgent (Phase 1),
    validée/ajustée pendant la session MIRROR (Phase 2),
    monitorée par les agents (Phase 3+).
    """

    friction_id: str = Field(
        ...,
        description="Identifiant unique (ex: 'F1', 'F2')",
    )
    department: str = Field(
        ...,
        description="Département concerné : 'sales', 'ops', 'finance', 'marketing'",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Titre court et factuel (ex: 'Pipeline gonflé de 3.57x')",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Description détaillée de la friction",
    )

    # ── Chiffrage ──
    estimated_annual_cost: float = Field(
        ...,
        ge=0,
        description="Coût annuel estimé en euros",
    )
    cost_confidence: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Confiance dans l'estimation. < 0.5 = estimation grossière.",
    )
    evidence: str = Field(
        ...,
        min_length=1,
        description="Les données qui justifient le chiffrage",
    )

    # ── Classification ──
    severity: FrictionSeverity
    impact_score: float = Field(
        ...,
        ge=0,
        le=10,
        description="Impact sur le business (0-10)",
    )
    difficulty_score: float = Field(
        ...,
        ge=0,
        le=10,
        description="Difficulté de résolution (0-10). 0 = trivial, 10 = transformation.",
    )
    priority: PriorityQuadrant

    # ── Résolution ──
    agent_recommended: Optional[str] = Field(
        default=None,
        description="Type d'agent recommandé pour cette friction",
    )
    resolution_status: str = Field(
        default="identified",
        description="'identified', 'in_progress', 'resolved', 'accepted'",
    )
    resolution_notes: Optional[str] = None

    # ── Méta ──
    identified_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    # ── Validators ──

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        allowed = {"sales", "ops", "finance", "marketing", "general"}
        if v not in allowed:
            raise ValueError(f"department must be one of {allowed}")
        return v

    @field_validator("resolution_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"identified", "in_progress", "resolved", "accepted"}
        if v not in allowed:
            raise ValueError(f"resolution_status must be one of {allowed}")
        return v

    @field_validator("severity", mode="before")
    @classmethod
    def auto_severity_from_cost(cls, v, info):
        """Si severity n'est pas fourni explicitement,
        on peut le déduire du coût. Mais ici on force l'explicite."""
        return v

    # ── Méthodes ──

    @staticmethod
    def classify_severity(cost: float) -> FrictionSeverity:
        """Méthode utilitaire pour classifier à partir du coût."""
        if cost >= 250_000:
            return FrictionSeverity.CRITICAL
        if cost >= 100_000:
            return FrictionSeverity.HIGH
        if cost >= 25_000:
            return FrictionSeverity.MEDIUM
        return FrictionSeverity.LOW

    @staticmethod
    def classify_priority(impact: float, difficulty: float) -> PriorityQuadrant:
        """Matrice impact × difficulté.
        Seuil impact : 5/10. Seuil difficulté : 5/10."""
        high_impact = impact >= 5
        easy = difficulty < 5

        if high_impact and easy:
            return PriorityQuadrant.QUICK_WIN
        if high_impact and not easy:
            return PriorityQuadrant.STRATEGIC
        if not high_impact and easy:
            return PriorityQuadrant.NICE_TO_HAVE
        return PriorityQuadrant.IGNORE


class FrictionMap(BaseModel):
    """La Friction Map complète d'un client.
    Construite pendant la session MIRROR, affinée par l'orchestrateur."""

    company_id: str = Field(..., min_length=1)
    frictions: list[Friction] = Field(
        ...,
        min_length=1,
        description="Au moins une friction doit être identifiée",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    validated_by_client: bool = Field(
        default=False,
        description="True si le CEO a validé la friction map pendant la session",
    )

    # ── Propriétés calculées ──

    @property
    def total_annual_cost(self) -> float:
        return sum(f.estimated_annual_cost for f in self.frictions)

    @property
    def quick_wins(self) -> list[Friction]:
        return [
            f for f in self.frictions
            if f.priority == PriorityQuadrant.QUICK_WIN
        ]

    @property
    def strategic(self) -> list[Friction]:
        return [
            f for f in self.frictions
            if f.priority == PriorityQuadrant.STRATEGIC
        ]

    @property
    def by_department(self) -> dict[str, list[Friction]]:
        result: dict[str, list[Friction]] = {}
        for f in self.frictions:
            result.setdefault(f.department, []).append(f)
        return result

    @property
    def by_severity(self) -> dict[FrictionSeverity, list[Friction]]:
        result: dict[FrictionSeverity, list[Friction]] = {}
        for f in self.frictions:
            result.setdefault(f.severity, []).append(f)
        return result

    @property
    def sorted_by_cost(self) -> list[Friction]:
        """Frictions triées par coût décroissant."""
        return sorted(
            self.frictions,
            key=lambda f: f.estimated_annual_cost,
            reverse=True,
        )

    @property
    def top_3(self) -> list[Friction]:
        """Les 3 frictions les plus coûteuses."""
        return self.sorted_by_cost[:3]

    @property
    def avg_confidence(self) -> float:
        """Confiance moyenne sur les estimations de coût."""
        if not self.frictions:
            return 0.0
        return sum(f.cost_confidence for f in self.frictions) / len(self.frictions)

    def format_table(self) -> str:
        """Format texte de la friction map pour le rapport."""
        lines = [
            f"{'FRICTION':<30} {'COÛT/AN':>10} {'DEPT':<10} {'PRIORITÉ':<12}",
            "─" * 65,
        ]
        for f in self.sorted_by_cost:
            cost_str = f"{f.estimated_annual_cost:,.0f}€"
            lines.append(
                f"{f.title:<30} {cost_str:>10} {f.department:<10} {f.priority.value:<12}"
            )
        lines.append("─" * 65)
        lines.append(
            f"{'TOTAL':<30} {self.total_annual_cost:>10,.0f}€"
        )
        return "\n".join(lines)
