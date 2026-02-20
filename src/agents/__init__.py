"""
KURIA Agents — Couche d'analyse et d'intelligence.

Chaque agent hérite de BaseAgent et implémente le même contrat.
Chaque agent est calibré par l'orchestrateur selon le profil client.
Chaque agent publie des événements consommés par le router.

Agents V1 :
- ScannerAgent         → Diagnostic initial (Phase 1)
- RevenueVelocityAgent → Pipeline truth + lead scoring
- ProcessClarityAgent  → Bottleneck detection + waste
- CashPredictabilityAgent → Cash forecast + alertes
- AcquisitionEfficiencyAgent → CAC + channel scoring
"""

from agents.base import BaseAgent, AgentError, AgentResult
from agents.scanner import ScannerAgent
from agents.revenue_velocity import RevenueVelocityAgent
from agents.process_clarity import ProcessClarityAgent
from agents.cash_predictability import CashPredictabilityAgent
from agents.acquisition_efficiency import AcquisitionEfficiencyAgent

__all__ = [
    "BaseAgent",
    "AgentError",
    "AgentResult",
    "ScannerAgent",
    "RevenueVelocityAgent",
    "ProcessClarityAgent",
    "CashPredictabilityAgent",
    "AcquisitionEfficiencyAgent",
]
