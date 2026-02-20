"""
KURIA Models — Schémas de données validés (Pydantic v2)

Chaque modèle est la SOURCE DE VÉRITÉ pour sa structure.
Aucun dict libre ne traverse le système.
Si une donnée existe, elle a un modèle.
"""

from models.company import (
    CompanyProfile,
    CompanySize,
    GrowthStage,
    ToolConnection,
    ToolProvider,
    ConnectedTools,
)
from models.data_portrait import (
    DataPortrait,
    SalesAnalysis,
    OperationsAnalysis,
    FinanceAnalysis,
    MarketingAnalysis,
    Anomaly,
    AnomalyType,
)
from models.friction import (
    Friction,
    FrictionMap,
    FrictionSeverity,
    PriorityQuadrant,
)
from models.clarity_score import (
    ClarityScore,
    DepartmentScores,
)
from models.events import (
    Event,
    EventType,
    EventPriority,
)
from models.agent_config import (
    AgentType,
    BaseAgentConfig,
    RevenueVelocityConfig,
    ProcessClarityConfig,
    CashPredictabilityConfig,
    AcquisitionEfficiencyConfig,
    AgentConfigSet,
)
from models.metrics import (
    AgentMetrics,
    RevenueVelocityMetrics,
    ProcessClarityMetrics,
    CashPredictabilityMetrics,
    AcquisitionEfficiencyMetrics,
    MetricTrend,
)
from models.report import (
    ReportSection,
    WeeklyReport,
    AttentionItem,
    AttentionLevel,
)

__all__ = [
    # Company
    "CompanyProfile",
    "CompanySize",
    "GrowthStage",
    "ToolConnection",
    "ToolProvider",
    "ConnectedTools",
    # Data Portrait
    "DataPortrait",
    "SalesAnalysis",
    "OperationsAnalysis",
    "FinanceAnalysis",
    "MarketingAnalysis",
    "Anomaly",
    "AnomalyType",
    # Friction
    "Friction",
    "FrictionMap",
    "FrictionSeverity",
    "PriorityQuadrant",
    # Clarity Score
    "ClarityScore",
    "DepartmentScores",
    # Events
    "Event",
    "EventType",
    "EventPriority",
    # Agent Config
    "AgentType",
    "BaseAgentConfig",
    "RevenueVelocityConfig",
    "ProcessClarityConfig",
    "CashPredictabilityConfig",
    "AcquisitionEfficiencyConfig",
    "AgentConfigSet",
    # Metrics
    "AgentMetrics",
    "RevenueVelocityMetrics",
    "ProcessClarityMetrics",
    "CashPredictabilityMetrics",
    "AcquisitionEfficiencyMetrics",
    "MetricTrend",
    # Report
    "ReportSection",
    "WeeklyReport",
    "AttentionItem",
    "AttentionLevel",
]
