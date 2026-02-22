"""
Ads Adapter — Google Ads, Meta Ads.

Gestion des campagnes publicitaires.
TOUJOURS risk_level B (validation humaine obligatoire).
En V1 : lecture seule + recommandations.
"""

from __future__ import annotations

from typing import Any

from models.action import Action
from services.config import get_settings


class AdsAdapter:
    """
    Adapter pour les plateformes publicitaires.

    V1 : Recommandations seulement (pas d'exécution directe).
    Les actions sont loggées pour approbation humaine.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ──────────────────────────────────────────────────────
    # CHANNEL MANAGEMENT
    # ──────────────────────────────────────────────────────

    async def pause_channel(self, action: Action) -> dict[str, Any]:
        """
        Pause un canal publicitaire.

        TOUJOURS risk B — jamais exécuté automatiquement.
        Retourne la recommandation pour approbation.
        """
        channel = action.target
        reason = action.parameters.get("reason", "")
        current_spend = action.parameters.get("current_spend", 0)

        return {
            "action": "pause_channel",
            "channel": channel,
            "current_spend": current_spend,
            "reason": reason,
            "executed": False,
            "recommendation": (
                f"Recommandation : mettre en pause le canal '{channel}' "
                f"(dépense actuelle : {current_spend}€/mois). "
                f"Raison : {reason}"
            ),
            "requires_manual_action": True,
            "instructions": [
                f"1. Se connecter à la plateforme {channel}",
                f"2. Mettre en pause les campagnes actives",
                f"3. Confirmer dans Kuria",
            ],
        }

    async def scale_channel(self, action: Action) -> dict[str, Any]:
        """
        Scale up un canal publicitaire.

        TOUJOURS risk B — jamais exécuté automatiquement.
        """
        channel = action.target
        current_spend = action.parameters.get("current_spend", 0)
        recommended_spend = action.parameters.get("recommended_spend", 0)
        reason = action.parameters.get("reason", "")

        return {
            "action": "scale_channel",
            "channel": channel,
            "current_spend": current_spend,
            "recommended_spend": recommended_spend,
            "increase_pct": (
                round(
                    (recommended_spend - current_spend) / max(current_spend, 1) * 100,
                    1,
                )
                if current_spend > 0
                else 0
            ),
            "reason": reason,
            "executed": False,
            "recommendation": (
                f"Recommandation : augmenter le budget '{channel}' "
                f"de {current_spend}€ à {recommended_spend}€/mois "
                f"(+{recommended_spend - current_spend}€). "
                f"Raison : {reason}"
            ),
            "requires_manual_action": True,
            "instructions": [
                f"1. Se connecter à la plateforme {channel}",
                f"2. Ajuster le budget à {recommended_spend}€/mois",
                f"3. Confirmer dans Kuria",
            ],
        }

    async def adjust_budget(self, action: Action) -> dict[str, Any]:
        """
        Ajuste le budget entre canaux.

        TOUJOURS risk B.
        Retourne le plan de réallocation complet.
        """
        adjustments = action.parameters.get("adjustments", [])
        total_budget = action.parameters.get("total_budget", 0)

        plan = []
        for adj in adjustments:
            channel = adj.get("channel", "")
            current = adj.get("current_spend", 0)
            recommended = adj.get("recommended_spend", 0)
            delta = recommended - current

            direction = "increase" if delta > 0 else "decrease" if delta < 0 else "maintain"

            plan.append({
                "channel": channel,
                "current_spend": current,
                "recommended_spend": recommended,
                "delta": delta,
                "direction": direction,
                "reason": adj.get("reason", ""),
            })

        return {
            "action": "adjust_budget",
            "total_budget": total_budget,
            "channels_affected": len(plan),
            "plan": plan,
            "executed": False,
            "requires_manual_action": True,
            "summary": self._format_budget_summary(plan),
        }

    @staticmethod
    def _format_budget_summary(plan: list[dict[str, Any]]) -> str:
        """Formate le plan de réallocation en texte lisible."""
        lines = ["📊 Plan de réallocation budget :", ""]

        cuts = [p for p in plan if p["direction"] == "decrease"]
        maintains = [p for p in plan if p["direction"] == "maintain"]
        increases = [p for p in plan if p["direction"] == "increase"]

        if cuts:
            lines.append("🔴 RÉDUIRE :")
            for p in cuts:
                lines.append(
                    f"  • {p['channel']} : {p['current_spend']}€ → "
                    f"{p['recommended_spend']}€ ({p['delta']:+.0f}€)"
                )

        if maintains:
            lines.append("🟡 MAINTENIR :")
            for p in maintains:
                lines.append(f"  • {p['channel']} : {p['current_spend']}€")

        if increases:
            lines.append("🟢 AUGMENTER :")
            for p in increases:
                lines.append(
                    f"  • {p['channel']} : {p['current_spend']}€ → "
                    f"{p['recommended_spend']}€ ({p['delta']:+.0f}€)"
                )

        return "\n".join(lines)
