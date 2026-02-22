"""
KURIA Agents — LLM-first, Decision Engine.

5 agents, même pattern :
  Trigger → State → LLM → Decision → Validate → Execute → Log

Agent 0 (Scanner) tourne tous les 3 jours — consulting permanent.
Agents 1-4 tournent selon leur fréquence configurée.
"""

from agents.base import BaseAgent
from agents.scanner import ScannerAgent, ScanMode
from agents.revenue_velocity import RevenueVelocityAgent
from agents.process_clarity import ProcessClarityAgent
from agents.cash_predictability import CashPredictabilityAgent
from agents.acquisition_efficiency import AcquisitionEfficiencyAgent

__all__ = [
    "BaseAgent",
    "ScannerAgent",
    "ScanMode",
    "RevenueVelocityAgent",
    "ProcessClarityAgent",
    "CashPredictabilityAgent",
    "AcquisitionEfficiencyAgent",
]
