"""
ClarityScore — Le Score de Clarté™.

Deux composantes :
1. Machine Readability  → Les agents peuvent-ils LIRE l'entreprise ?
2. Structural Compatibility → L'entreprise peut-elle INTÉGRER des agents ?

Ce score est la VARIABLE D'ENTRÉE qui détermine
la qualité de tout ce que les agents produisent.

Design decisions :
- Score 0-100 pour chaque composante et pour le global
- Le global N'EST PAS une moyenne — c'est une moyenne pondérée
  avec plus de poids sur machine_readability (c'est le facteur limitant)
- Scores par département pour l'orchestrateur
- Confidence level pour indiquer la fiabilité du score lui-même
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DepartmentScores(BaseModel):
    """Scores de clarté par département.
    Utilisé par l'orchestrateur pour décider quels agents prioriser
    et comment les configurer."""

    sales: float = Field(..., ge=0, le=100)
    ops: float = Field(..., ge=0, le=100)
    finance: float = Field(..., ge=0, le=100)
    marketing: float = Field(..., ge=0, le=100)

    @property
    def weakest_department(self) -> str:
        """Le département le plus faible = la priorité d'action."""
        scores = {
            "sales": self.sales,
            "ops": self.ops,
            "finance": self.finance,
            "marketing": self.marketing,
        }
        return min(scores, key=scores.get)

    @property
    def strongest_department(self) -> str:
        scores = {
            "sales": self.sales,
            "ops": self.ops,
            "finance": self.finance,
            "marketing": self.marketing,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict[str, float]:
        return {
            "sales": self.sales,
            "ops": self.ops,
            "finance": self.finance,
            "marketing": self.marketing,
        }


class ClarityScore(BaseModel):
    """Le Score de Clarté™ complet.

    Calculé par le ScannerAgent lors de la Phase 1.
    Recalculé chaque semaine par l'Adaptateur.
    Affiché en gros dans le dashboard et le rapport du lundi.
    """

    # ── Composantes ──
    machine_readability: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "À quel point les agents IA peuvent LIRE l'entreprise. "
            "Dépend de : qualité des données, complétude, structure, accessibilité."
        ),
    )
    structural_compatibility: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "À quel point l'organisation peut INTÉGRER des agents. "
            "Dépend de : données centralisées, process explicites, rôles clairs."
        ),
    )

    # ── Score global ──
    overall: float = Field(
        ...,
        ge=0,
        le=100,
        description="Score global calculé. PAS une moyenne simple.",
    )

    # ── Détail par département ──
    by_department: DepartmentScores

    # ── Fiabilité ──
    confidence: float = Field(
        ...,
        ge=0,
        le=1.0,
        description=(
            "Confiance dans le score lui-même. "
            "Bas si peu de données, peu d'outils connectés, ou scan partiel."
        ),
    )
    data_sources_used: int = Field(
        ...,
        ge=0,
        description="Nombre de sources de données utilisées pour calculer le score.",
    )

    # ── Historique ──
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    previous_overall: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Score overall de la semaine précédente (pour le trend).",
    )

    # ── Validators ──

    @model_validator(mode="after")
    def validate_overall_calculation(self) -> "ClarityScore":
        """Vérifie que le score overall est cohérent avec les composantes.
        Formule : 60% machine_readability + 40% structural_compatibility.
        Machine readability pèse plus car c'est le facteur limitant :
        si les agents ne peuvent pas lire, rien ne marche."""
        expected = (
            self.machine_readability * 0.6
            + self.structural_compatibility * 0.4
        )
        tolerance = 1.0  # Tolérance pour arrondis
        if abs(self.overall - expected) > tolerance:
            raise ValueError(
                f"overall ({self.overall}) incohérent avec les composantes. "
                f"Attendu : {expected:.1f} "
                f"(0.6 × {self.machine_readability} + 0.4 × {self.structural_compatibility})"
            )
        return self

    # ── Méthodes ──

    @staticmethod
    def calculate_overall(
        machine_readability: float,
        structural_compatibility: float,
    ) -> float:
        """Méthode de calcul centralisée.
        Utilisée par le ScannerAgent et l'Adaptateur."""
        return round(machine_readability * 0.6 + structural_compatibility * 0.4, 1)

    @property
    def trend(self) -> Optional[float]:
        """Variation vs semaine précédente. Positif = amélioration."""
        if self.previous_overall is None:
            return None
        return round(self.overall - self.previous_overall, 1)

    @property
    def trend_emoji(self) -> str:
        """Pour le rapport du lundi."""
        t = self.trend
        if t is None:
            return "—"
        if t > 2:
            return "↑"
        if t < -2:
            return "↓"
        return "→"

    @property
    def grade(self) -> str:
        """Classification lisible du score.
        Utilisé dans le rapport et le dashboard."""
        if self.overall >= 80:
            return "EXCELLENT"
        if self.overall >= 60:
            return "BON"
        if self.overall >= 40:
            return "INSUFFISANT"
        if self.overall >= 20:
            return "CRITIQUE"
        return "OPAQUE"

    def format_for_email(self) -> str:
        """Ligne du rapport du lundi."""
        trend_str = ""
        if self.trend is not None:
            sign = "+" if self.trend >= 0 else ""
            trend_str = f" ({self.trend_emoji}{sign}{self.trend})"
        return f"SCORE DE CLARTÉ : {self.overall:.0f}/100{trend_str} — {self.grade}"

    class Config:
        json_schema_extra = {
            "example": {
                "machine_readability": 29,
                "structural_compatibility": 39,
                "overall": 33.0,
                "by_department": {
                    "sales": 41,
                    "ops": 22,
                    "finance": 48,
                    "marketing": 27,
                },
                "confidence": 0.72,
                "data_sources_used": 3,
                "previous_overall": 30.0,
            }
  }
