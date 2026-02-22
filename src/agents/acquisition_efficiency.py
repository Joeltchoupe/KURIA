"""
Agent 4 : Acquisition Efficiency — LLM-first.

KPI : Blended CAC (€/client acquis).

Actions :
  4.1 — Attribution [A] (déterministe)
  4.3 — Budget Reallocation [B] (LLM + validation humaine)
  4.x — Channel Analysis [A]

Souvent DÉSACTIVÉ en V1 si données insuffisantes.

System prompt : prompts/system/acquisition_efficiency.txt
User prompts : prompts/acquisition_efficiency/*.txt
"""

from __future__ import annotations

from typing import Any

from models.event import Event
from models.state import StateSnapshot, MarketingState, ChannelState
from models.decision import Decision, DecisionType, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class AcquisitionEfficiencyAgent(BaseAgent):
    """
    Acquisition Efficiency — Le LLM optimise vos canaux.

    Attribution = déterministe (first/last/linear).
    Analyse qualitative + budget reallocation = LLM.
    Budget changes = toujours Risk B (validation humaine).
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ACQUISITION_EFFICIENCY

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques injectées dans les prompts Acquisition."""
        params = self.config.parameters
        return {
            "attribution_model": params.get("attribution_model", "positional"),
            "cac_anomaly_threshold_pct": params.get(
                "cac_anomaly_threshold_pct", 0.25
            ),
            "min_leads_per_channel": params.get("min_leads_per_channel", 5),
            "min_clients_for_analysis": params.get(
                "min_clients_for_analysis", 10
            ),
            "first_touch_weight": params.get("first_touch_weight", 0.3),
            "last_touch_weight": params.get("last_touch_weight", 0.3),
        }

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Construit le snapshot marketing.

        Les calculs déterministes (blended CAC, LTV/CAC ratio)
        sont faits ICI, pas par le LLM.
        """
        marketing = None

        if raw_data and "channels" in raw_data:
            channels: list[ChannelState] = []

            for ch in raw_data["channels"]:
                channels.append(ChannelState(
                    channel=ch.get("channel", ""),
                    spend_30d=ch.get("spend_30d", 0),
                    leads_30d=ch.get("leads_30d", 0),
                    clients_30d=ch.get("clients_30d", 0),
                    cac=ch.get("cac", 0),
                    conversion_rate=ch.get("conversion_rate", 0),
                    trend_pct=ch.get("trend_pct", 0),
                ))

            # Calculs déterministes
            total_spend = sum(ch.spend_30d for ch in channels)
            total_leads = sum(ch.leads_30d for ch in channels)
            total_clients = sum(ch.clients_30d for ch in channels)
            blended_cac = (
                round(total_spend / total_clients, 2)
                if total_clients > 0
                else 0
            )
            avg_ltv = raw_data.get("avg_ltv", 0)
            ltv_cac_ratio = (
                round(avg_ltv / blended_cac, 2)
                if blended_cac > 0
                else 0
            )

            # Calculer CAC par channel si non fourni
            for ch in channels:
                if ch.cac == 0 and ch.clients_30d > 0:
                    ch.cac = round(ch.spend_30d / ch.clients_30d, 2)
                if ch.conversion_rate == 0 and ch.leads_30d > 0:
                    ch.conversion_rate = round(
                        ch.clients_30d / ch.leads_30d, 4
                    )

            marketing = MarketingState(
                channels=channels,
                total_leads_30d=total_leads,
                total_spend_30d=total_spend,
                blended_cac=blended_cac,
                avg_ltv=avg_ltv,
                ltv_cac_ratio=ltv_cac_ratio,
            )

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:30]]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            marketing=marketing,
            recent_events=recent,
            previous_decisions=[
                {
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:200],
                }
                for d in self._decisions[-5:]
            ],
        )

    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide les règles métier Acquisition.

        Règles :
          - Budget changes = TOUJOURS Risk B
          - Pas de modification pub directe
          - Données suffisantes requises
        """
        errors: list[str] = []

        for action in decision.actions:
            # Budget = TOUJOURS approval humaine
            if action.action in (
                "adjust_budget",
                "pause_channel",
                "scale_channel",
            ):
                if action.risk_level != RiskLevel.B:
                    # Force B plutôt que bloquer
                    action.risk_level = RiskLevel.B

            # Pas d'exécution pub directe
            if action.action in (
                "create_ad",
                "modify_ad",
                "delete_ad",
                "create_campaign",
                "delete_campaign",
            ):
                errors.append(
                    f"INTERDIT : modification pub directe '{action.action}'"
                )

            # Budget négatif
            if action.action == "adjust_budget":
                adjustments = action.parameters.get("adjustments", [])
                for adj in adjustments:
                    if isinstance(adj, dict):
                        rec = adj.get("recommended_spend", 0)
                        if rec < 0:
                            errors.append(
                                f"Budget négatif recommandé pour "
                                f"'{adj.get('channel', '?')}' : {rec}"
                            )

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.ANALYZE_CHANNELS

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS — Raccourcis par action
    # ──────────────────────────────────────────────────────

    async def analyze_channels(
        self,
        channels_data: list[dict[str, Any]],
        avg_ltv: float = 0,
    ) -> Decision:
        """
        Action 4.x — Channel Analysis [A].

        Analyse chaque canal : CAC, LTV/CAC, trend.
        Verdict par canal : SCALE / MAINTAIN / MONITOR / CUT.
        """
        return await self.run(
            raw_data={
                "channels": channels_data,
                "avg_ltv": avg_ltv,
            },
            prompt_name="acquisition_efficiency/channel_analysis",
        )

    async def reallocate_budget(
        self,
        channels_data: list[dict[str, Any]],
        avg_ltv: float = 0,
        total_budget: float | None = None,
    ) -> Decision:
        """
        Action 4.3 — Budget Reallocation [B].

        Risk B : toute recommandation budget nécessite validation humaine.
        Catégorise en CUT / MAINTAIN / DOUBLE.
        """
        total = total_budget or sum(
            ch.get("spend_30d", 0) for ch in channels_data
        )

        return await self.run(
            raw_data={
                "channels": channels_data,
                "avg_ltv": avg_ltv,
                "total_budget": total,
            },
            prompt_name="acquisition_efficiency/budget_reallocation",
        )

    async def score_lead_quality(
        self,
        leads_data: list[dict[str, Any]],
        channels_data: list[dict[str, Any]],
    ) -> Decision:
        """
        Score la qualité des leads par canal.

        Pas de prompt dédié — utilise le channel_analysis
        avec un focus lead quality.
        """
        return await self.run(
            raw_data={
                "channels": channels_data,
                "leads": leads_data,
            },
            prompt_name="acquisition_efficiency/channel_analysis",
            prompt_variables={"focus": "lead_quality"},
  )
