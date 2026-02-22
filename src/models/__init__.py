"""
KURIA Models — LLM-First Architecture.

Core (universel) :
  Event, StateSnapshot, Decision, Action, ActionLog

Business :
  CompanyProfile, ClarityScore, WeeklyReport

Config :
  AgentConfig, Threshold, *Config, AgentConfigSet
"""

from models.event import Event, EventType
from models.state import (
    StateSnapshot,
    DealState,
    TaskState,
    CashState,
    MarketingState,
    ChannelState,
)
from models.decision import (
    Decision,
    DecisionType,
    RiskLevel,
    ActionRequest,
)
from models.action import Action, ActionStatus, ActionLog
from models.company import CompanyProfile, CompanySize, GrowthStage
from models.clarity_score import ClarityScore
from models.report import (
    WeeklyReport,
    KPIStatus,
    DecisionSummary,
    PendingApproval,
    AdaptationNote,
    AttentionLevel,
)
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
    # Core
    "Event",
    "EventType",
    "StateSnapshot",
    "DealState",
    "TaskState",
    "CashState",
    "MarketingState",
    "ChannelState",
    "Decision",
    "DecisionType",
    "RiskLevel",
    "ActionRequest",
    "Action",
    "ActionStatus",
    "ActionLog",
    # Business
    "CompanyProfile",
    "CompanySize",
    "GrowthStage",
    "ClarityScore",
    "WeeklyReport",
    "KPIStatus",
    "DecisionSummary",
    "PendingApproval",
    "AdaptationNote",
    "AttentionLevel",
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
