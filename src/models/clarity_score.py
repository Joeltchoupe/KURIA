"""
ClarityScore — Le Score de Clarté™.

UN chiffre. 0-100. Deux composantes :

1. LISIBILITÉ MACHINE
   → À quel point les agents IA peuvent lire l'entreprise
   → Plus c'est bas, plus les outputs sont mauvais

2. COMPATIBILITÉ STRUCTURELLE
   → À quel point l'organisation peut intégrer des agents
   → Données centralisées, process explicites, rôles clairs

Ce n'est pas un gadget. C'est la VARIABLE D'ENTRÉE
qui détermine la qualité de tout le reste.

Simplifié : le calcul est fait par le LLM.
Ce modèle stocke le RÉSULTAT.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ClarityScore(BaseModel):
    """
    Score de Clarté d'une entreprise.

    Calculé au scan initial, mis à jour chaque semaine.
    C'est la baseline pour mesurer la progression.
    """
    company_id: str
    score: float = Field(..., ge=0, le=100, description="Score global 0-100")

    # Composantes
    machine_readability: float = Field(
        ..., ge=0, le=100,
        description="Les agents peuvent-ils lire les données ?",
    )
    structural_compatibility: float = Field(
        ..., ge=0, le=100,
        description="L'organisation peut-elle intégrer des agents ?",
    )

    # Sous-scores (optionnels, remplis par le LLM)
    data_quality: float = Field(default=0, ge=0, le=100)
    data_completeness: float = Field(default=0, ge=0, le=100)
    process_explicitness: float = Field(default=0, ge=0, le=100)
    tool_integration: float = Field(default=0, ge=0, le=100)

    # Contexte
    reasoning: str = Field(
        default="",
        description="Explication du score par le LLM",
    )
    top_blockers: list[str] = Field(
        default_factory=list,
        description="Les 3 plus gros freins à la clarté",
    )
    quick_wins: list[str] = Field(
        default_factory=list,
        description="Actions rapides pour monter le score",
    )

    # Méta
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    previous_score: float | None = Field(default=None, ge=0, le=100)
    version: int = Field(default=1, ge=1)

    @property
    def trend(self) -> str:
        """Tendance vs le score précédent."""
        if self.previous_score is None:
            return "initial"
        delta = self.score - self.previous_score
        if delta > 3:
            return "improving"
        if delta < -3:
            return "degrading"
        return "stable"

    @property
    def health(self) -> str:
        """Santé globale."""
        if self.score >= 70:
            return "healthy"
        if self.score >= 40:
            return "warning"
        return "critical"

    @property
    def summary(self) -> dict[str, Any]:
        """Résumé pour le dashboard et les rapports."""
        return {
            "score": self.score,
            "health": self.health,
            "trend": self.trend,
            "machine_readability": self.machine_readability,
            "structural_compatibility": self.structural_compatibility,
            "top_blockers": self.top_blockers[:3],
            "quick_wins": self.quick_wins[:3],
  }
