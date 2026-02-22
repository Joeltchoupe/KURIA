"""
KURIA Orchestrator — Le système nerveux central.

L'orchestrateur est LE MOAT de Kuria.
Même agents. Comportement différent. Parce que chaque entreprise
est différente. Et l'orchestrateur le sait.

5 modules :
  - profile.py      → Scan → CompanyProfile → AgentConfigSet
  - engine.py       → Dispatch, scheduling, run loop
  - coordinator.py  → Signaux inter-agents
  - adapter.py      → Recalibration hebdo
  - reporter.py     → Compilation → WeeklyReport
"""

from orchestrator.engine import OrchestrationEngine, RunResult, AgentRunStatus
from orchestrator.profile import ProfileGenerator
from orchestrator.coordinator import AgentCoordinator, Signal, SignalType
from orchestrator.adapter import ConfigAdapter, AdaptationResult
from orchestrator.reporter import WeeklyReporter

__all__ = [
    "OrchestrationEngine",
    "RunResult",
    "AgentRunStatus",
    "ProfileGenerator",
    "AgentCoordinator",
    "Signal",
    "SignalType",
    "ConfigAdapter",
    "AdaptationResult",
    "WeeklyReporter",
]
