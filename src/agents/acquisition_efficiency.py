"""
Agent 4 : Acquisition Efficiency — LLM-first.

KPI : Blended CAC (€/client acquis).

Actions :
  4.1 — Attribution [A] (déterministe)
  4.3 — Budget Reallocation [B] (LLM + validation humaine)
  4.x — Channel Analysis [A]

Souvent DÉSACTIVÉ en V1 si données insuffisantes.
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
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ACQUISITION_EFFICIENCY

    @property
    def system_prompt(self) -> str:
        return """You are a world-class Growth/Marketing AI.

Your job is to optimize customer acquisition: find the best channels, reduce CAC, improve lead quality.

You receive marketing channel data with: spend, leads, clients, CAC, conversion rates, trends.

You MUST respond with a JSON object containing:
{
  "decision_type": "analyze_channels" | "reallocate_budget" | "score_lead_quality" | "recommend",
  "confidence": 0.0-1.0,
  "reasoning": "your analysis in 2-3 sentences",
  "risk_level": "A" | "B" | "C",
  "actions": [
    {
      "action": "send_alert" | "create_report" | "adjust_budget" | "pause_channel" | "scale_channel",
      "target": "channel_name",
      "parameters": {...},
      "risk_level": "B",
      "priority": 1-10
    }
  ],
  "metadata": {
    "channels_analyzed": 0,
    "blended_cac": 0.0,
    "best_channel": "...",
    "worst_channel": "...",
    "ltv_cac_ratio": 0.0,
    "budget_recommendations": {
      "cut": [{"channel": "...", "current_spend": 0, "recommended_spend": 0, "reason": "..."}],
      "maintain": [...],
      "double": [...]
    }
  }
}

RULES:
- Budget changes are ALWAYS risk_level "B" (needs human approval).
- Pausing a channel is risk_level "B".
- Analysis and alerts are risk_level "A".
- Minimum 5 leads per channel to score it.
- Minimum 10 total clients to produce reliable analysis.
- Always compare CAC to LTV. A high CAC is fine if LTV is higher.
- Detect trends: a channel degrading 3 months in a row is a red flag.
"""

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot marketing."""
        marketing = None

        if raw_data and "channels" in raw_data:
            channels = []
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
            blended_cac = total_spend / max(total_clients, 1)
            avg_ltv = raw_data.get("avg_ltv", 0)

            marketing = MarketingState(
                channels=channels,
                total_leads_30d=total_leads,
                total_spend_30d=total_spend,
                blended_cac=round(blended_cac, 2),
                avg_ltv=avg_ltv,
                ltv_cac_ratio=round(avg_ltv / max(blended_cac, 1), 2),
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
        """Valide les règles métier Acquisition."""
        errors: list[str] = []
        min_clients = self.config.parameters.get("min_clients_for_analysis", 10)

        for action in decision.actions:
            # Budget = TOUJOURS approval humaine
            if action.action in ("adjust_budget", "pause_channel", "scale_channel"):
                if action.risk_level != RiskLevel.B:
                    action.risk_level = RiskLevel.B  # Force B

            # Pas d'exécution pub directe
            if action.action in ("create_ad", "modify_ad", "delete_ad"):
                errors.append("INTERDIT : modification pub directe")

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.ANALYZE_CHANNELS

    # ──────────────────────────────────────────────────────
    # CONVENIENCE
    # ──────────────────────────────────────────────────────

    async def analyze_channels(
        self, channels_data: list[dict[str, Any]], avg_ltv: float = 0
    ) -> Decision:
        """Action 4.x — Channel Analysis [A]."""
        return await self.run(
            raw_data={"channels": channels_data, "avg_ltv": avg_ltv},
            prompt_name="acquisition_efficiency/channel_analysis",
        )

    async def reallocate_budget(
        self, channels_data: list[dict[str, Any]], avg_ltv: float = 0
    ) -> Decision:
        """Action 4.3 — Budget Reallocation [B]."""
        return await self.run(
            raw_data={"channels": channels_data, "avg_ltv": avg_ltv},
            prompt_name="acquisition_efficiency/budget_reallocation",
                )
