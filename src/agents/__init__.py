"""
KURIA Agents — LLM-first, Decision Engine.

4 agents, même pattern :
  Trigger → State → LLM → Decision → Validate → Execute → Log

Le code est le plombier. Le LLM est le cerveau.
Les prompts sont la logique métier.
"""

from agents.base import BaseAgent
from agents.revenue_velocity import RevenueVelocityAgent
from agents.process_clarity import ProcessClarityAgent
from agents.cash_predictability import CashPredictabilityAgent
from agents.acquisition_efficiency import AcquisitionEfficiencyAgent

__all__ = [
    "BaseAgent",
    "RevenueVelocityAgent",
    "ProcessClarityAgent",
    "CashPredictabilityAgent",
    "AcquisitionEfficiencyAgent",
]
