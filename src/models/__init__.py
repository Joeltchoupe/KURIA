"""
KURIA Models — LLM-First Architecture.

5 modèles universels :
  Event       → événement brut entrant
  State       → état compressé envoyé au LLM
  Decision    → sortie JSON du LLM
  Action      → action à exécuter
  ActionLog   → trace de ce qui a été fait

+ modèles de config (inchangés)
+ modèles business légers
"""

from models.event import Event, EventType
from models.state import (
    StateSnapshot,
    DealState,
    TaskState,
    CashState,
    MarketingState,
)
from models.decision import Decision, DecisionType, RiskLevel
from models.action import Action, ActionStatus, ActionLog
from models.company import CompanyProfile, CompanySize, GrowthStage
from models.agent_config import (
    AgentType,
    Threshold,
    AgentConfig,
    BaseAgentConfig,
    ScoreWeights,
    RevenueVelocityConfig,
    ProcessClarityConfig,
    CashPredictabilityConfig,
    AcquisitionEfficiencyConfig,
    AgentConfigSet,
)

__all__ = [
    # Core — Universal
    "Event",
    "EventType",
    "StateSnapshot",
    "DealState",
    "TaskState",
    "CashState",
    "MarketingState",
    "Decision",
    "DecisionType",
    "RiskLevel",
    "Action",
    "ActionStatus",
    "ActionLog",
    # Company
    "CompanyProfile",
    "CompanySize",
    "GrowthStage",
    # Agent Config
    "AgentType",
    "Threshold",
    "AgentConfig",
    "BaseAgentConfig",
    "ScoreWeights",
    "RevenueVelocityConfig",
    "ProcessClarityConfig",
    "CashPredictabilityConfig",
    "AcquisitionEfficiencyConfig",
    "AgentConfigSet",
]
