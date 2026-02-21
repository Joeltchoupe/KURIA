"""
KURIA Connectors V2 — 37 connecteurs, 13 catégories.

Chaque connecteur hérite de BaseConnector.
Chaque connecteur normalise vers des structures standard.
Les agents ne savent JAMAIS quel outil est derrière.

Le ConnectorRegistry permet de :
- Découvrir dynamiquement les connecteurs disponibles
- Instancier le bon connecteur à partir du nom du provider
- Lister les catégories et outils supportés
"""

from connectors.base import (
    BaseConnector,
    ConnectorError,
    RateLimitError,
    AuthenticationError,
    ExtractionMetrics,
)
from connectors.registry import ConnectorRegistry
from connectors.csv_fallback import CSVConnector

__all__ = [
    "BaseConnector",
    "ConnectorError",
    "RateLimitError",
    "AuthenticationError",
    "ExtractionMetrics",
    "ConnectorRegistry",
    "CSVConnector",
]
