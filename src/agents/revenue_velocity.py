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

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REVENUE_VELOCITY

    # ❌ SUPPRIMÉ : plus de @property system_prompt hardcodé
    # Le base.py charge prompts/system/revenue_velocity.txt automatiquement

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques Revenue Velocity."""
        params = self.config.parameters
        weights = params.get("score_weights", {})
        return {
            "stagnation_threshold_days": params.get("stagnation_threshold_days", 21),
            "zombie_threshold_days": params.get("zombie_threshold_days", 45),
            "hot_lead_threshold": params.get("hot_lead_threshold", 75),
            "archive_threshold": params.get("archive_threshold", 20),
            "score_fit": weights.get("fit", 0.3),
            "score_activity": weights.get("activity", 0.4),
            "score_timing": weights.get("timing", 0.2),
            "score_source": weights.get("source", 0.1),
            "pipeline_realistic_factor": params.get("pipeline_realistic_factor", 0.5),
            "high_value_deal_threshold": params.get("high_value_deal_threshold", 10000),
            "forecast_correction_factor": params.get("forecast_correction_factor", 1.0),
        }

    async def build_snapshot(self, events=None, raw_data=None):
        # ... inchangé ...

    def validate_decision(self, decision):
        # ... inchangé ...

    def _default_decision_type(self):
        return DecisionType.CLASSIFY_DEAL

    async def clean_pipeline(self, deals_data):
        return await self.run(
            raw_data={"deals": deals_data},
            prompt_name="revenue_velocity/pipeline_cleaner",
        )

    async def score_leads(self, leads_data):
        return await self.run(
            raw_data={"deals": leads_data},
            prompt_name="revenue_velocity/lead_scoring",
        )

    async def analyze_win_loss(self, won_deals, lost_deals):
        return await self.run(
            raw_data={"deals": won_deals + lost_deals},
            prompt_name="revenue_velocity/win_loss_analysis",
            prompt_variables={
                "won_deals": won_deals,
                "lost_deals": lost_deals,
            },
        )

    async def generate_proposal(self, deal_data):
        return await self.run(
            raw_data={"deals": [deal_data]},
            prompt_name="revenue_velocity/proposal_generator",
)


                    
