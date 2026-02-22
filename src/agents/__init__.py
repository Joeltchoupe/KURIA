"""
KURIA Agents — Couche d'analyse et d'intelligence.

Contrat :
- Chaque agent hérite de BaseAgent[ConfigT]
- Chaque agent implémente _validate(), _execute(), _confidence()
- Point d'entrée unique : await agent.execute(data) → AgentResult
- Les événements sont bufferisés et flush après exécution
- L'orchestrateur calibre chaque agent via sa config typée

Agents V1 :
- ScannerAgent                → Diagnostic initial (Phase 1)
- RevenueVelocityAgent        → Pipeline truth + lead scoring
- ProcessClarityAgent         → Bottleneck detection + waste
- CashPredictabilityAgent     → Cash forecast + alertes
- AcquisitionEfficiencyAgent  → CAC + channel scoring
"""

from agents.base import BaseAgent, AgentError, AgentResult, InsufficientDataError, ConnectorDataError
from agents.scanner import ScannerAgent
from agents.revenue_velocity import RevenueVelocityAgent
from agents.process_clarity import ProcessClarityAgent
from agents.cash_predictability import CashPredictabilityAgent
from agents.acquisition_efficiency import AcquisitionEfficiencyAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    "AgentError",
    "InsufficientDataError",
    "ConnectorDataError",
    # Agents
    "ScannerAgent",
    "RevenueVelocityAgent",
    "ProcessClarityAgent",
    "CashPredictabilityAgent",
    "AcquisitionEfficiencyAgent",
]
