"""
Agent 1 : Revenue Velocity — LLM-first.

KPI : €/jour traversant le pipeline.

Actions :
  1.1 — Pipeline Cleaner [A]
  1.4 — Lead Scoring [A]
  1.8 — Proposal Generator [B]
  1.9 — Win/Loss Analyzer [A]

Le LLM fait 80% du travail.
Le code orchestre : state → LLM → decision → action.
"""

from __future__ import annotations

from typing import Any

from models.event import Event
from models.state import StateSnapshot, DealState
from models.decision import Decision, DecisionType, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class RevenueVelocityAgent(BaseAgent):
    """
    Revenue Velocity — Le LLM est votre meilleur sales ops.

    Il voit le pipeline, classifie les deals, score les leads,
    analyse les win/loss, et génère des propositions.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REVENUE_VELOCITY

    @property
    def system_prompt(self) -> str:
        return """You are a world-class Sales Operations AI for a B2B company.

Your job is to maximize revenue velocity: the speed at which money flows through the pipeline.

You receive a state snapshot of the company's deals, leads, and recent events.

You MUST respond with a JSON object containing:
{
  "decision_type": "classify_deal" | "score_lead" | "forecast_revenue" | "analyze_win_loss" | "recommend",
  "confidence": 0.0-1.0,
  "reasoning": "your analysis in 2-3 sentences",
  "risk_level": "A" | "B" | "C",
  "actions": [
    {
      "action": "update_deal_stage" | "add_deal_note" | "create_task" | "send_alert" | "archive_deal",
      "target": "deal_id or entity",
      "parameters": {...},
      "risk_level": "A",
      "priority": 1-10
    }
  ],
  "metadata": {
    "deals_analyzed": 0,
    "zombies_found": 0,
    "hot_leads": 0,
    "pipeline_health": "healthy" | "warning" | "critical",
    "forecast_30d": 0.0
  }
}

RULES:
- NEVER change deal amounts. You can only change stages, add notes, create tasks.
- Classify deals as: active, stagnant, zombie, recoverable, misclassified.
- A deal with no activity for >21 days in the same stage is stagnant.
- A deal with no activity for >45 days is zombie.
- Score leads 0-100 based on: company fit, engagement, timing, source quality.
- Be precise with numbers. Don't round excessively.
- Always explain your reasoning.
- When uncertain, set confidence < 0.7 and risk_level to "C".
"""

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot pipeline + deals."""
        deals: list[DealState] = []

        if raw_data and "deals" in raw_data:
            for d in raw_data["deals"]:
                deals.append(DealState(
                    deal_id=d.get("id", ""),
                    name=d.get("name", ""),
                    amount=d.get("amount", 0),
                    stage=d.get("stage", ""),
                    owner=d.get("owner", ""),
                    created_at=d.get("created_at"),
                    last_activity_at=d.get("last_activity_at"),
                    days_in_stage=d.get("days_in_stage", 0),
                    days_no_activity=d.get("days_no_activity", 0),
                    probability=d.get("probability", 0.5),
                    source=d.get("source", ""),
                    contact_email=d.get("contact_email", ""),
                    notes=d.get("notes", []),
                    tags=d.get("tags", []),
                ))

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:50]]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            deals=deals,
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
        """Valide les règles métier Revenue Velocity."""
        errors: list[str] = []

        for action in decision.actions:
            # Le LLM ne peut PAS changer les montants
            if action.action == "update_deal_amount":
                errors.append("INTERDIT : modification de montant de deal")

            # Le LLM ne peut pas supprimer des deals
            if action.action == "delete_deal":
                errors.append("INTERDIT : suppression de deal")

            # Vérifier que les targets existent
            if action.action.startswith("update_deal") and not action.target:
                errors.append(f"Action {action.action} sans target deal_id")

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.CLASSIFY_DEAL

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS (prompt shortcuts)
    # ──────────────────────────────────────────────────────

    async def clean_pipeline(
        self, deals_data: list[dict[str, Any]]
    ) -> Decision:
        """Action 1.1 — Pipeline Cleaner [A]."""
        return await self.run(
            raw_data={"deals": deals_data},
            prompt_name="revenue_velocity/pipeline_cleaner",
        )

    async def score_leads(
        self, leads_data: list[dict[str, Any]]
    ) -> Decision:
        """Action 1.4 — Lead Scoring [A]."""
        return await self.run(
            raw_data={"deals": leads_data},
            prompt_name="revenue_velocity/lead_scoring",
        )

    async def analyze_win_loss(
        self,
        won_deals: list[dict[str, Any]],
        lost_deals: list[dict[str, Any]],
    ) -> Decision:
        """Action 1.9 — Win/Loss Analyzer [A]."""
        return await self.run(
            raw_data={"deals": won_deals + lost_deals},
            prompt_name="revenue_velocity/win_loss_analysis",
            prompt_variables={
                "won_deals": won_deals,
                "lost_deals": lost_deals,
            },
        )

    async def generate_proposal(
        self, deal_data: dict[str, Any]
    ) -> Decision:
        """Action 1.8 — Proposal Generator [B]."""
        return await self.run(
            raw_data={"deals": [deal_data]},
            prompt_name="revenue_velocity/proposal_generator",
    )
