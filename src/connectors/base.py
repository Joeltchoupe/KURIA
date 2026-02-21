"""
BaseConnector — Classe abstraite pour tous les connecteurs Kuria.

V2 additions :
- CONNECTOR_CATEGORY pour le registry
- Normalized output types documentés
- Méthodes de discovery (list_available_data)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# EXCEPTIONS (inchangées)
# ──────────────────────────────────────────────


class ConnectorError(Exception):
    def __init__(
        self,
        connector_name: str,
        message: str,
        recoverable: bool = True,
        raw_error: Optional[Exception] = None,
    ):
        self.connector_name = connector_name
        self.message = message
        self.recoverable = recoverable
        self.raw_error = raw_error
        super().__init__(f"[{connector_name}] {message}")


class RateLimitError(ConnectorError):
    def __init__(
        self,
        connector_name: str,
        retry_after_seconds: int = 60,
        raw_error: Optional[Exception] = None,
    ):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            connector_name=connector_name,
            message=f"Rate limit hit. Retry after {retry_after_seconds}s.",
            recoverable=True,
            raw_error=raw_error,
        )


class AuthenticationError(ConnectorError):
    def __init__(
        self,
        connector_name: str,
        message: str = "Authentication failed",
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            connector_name=connector_name,
            message=message,
            recoverable=False,
            raw_error=raw_error,
        )


# ──────────────────────────────────────────────
# EXTRACTION METRICS
# ──────────────────────────────────────────────


class ExtractionMetrics(BaseModel):
    connector_name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_extracted: int = 0
    errors_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    status: str = "pending"

    def complete(self, records: int, errors: int = 0) -> None:
        self.completed_at = datetime.utcnow()
        self.duration_seconds = (
            self.completed_at - self.started_at
        ).total_seconds()
        self.records_extracted = records
        self.errors_count = errors
        self.status = "success" if errors == 0 else "partial"

    def fail(self, error_message: str) -> None:
        self.completed_at = datetime.utcnow()
        self.duration_seconds = (
            self.completed_at - self.started_at
        ).total_seconds()
        self.status = "failed"
        self.warnings.append(error_message)


# ──────────────────────────────────────────────
# BASE CONNECTOR
# ──────────────────────────────────────────────


class BaseConnector(ABC):
    """Classe abstraite pour tous les connecteurs Kuria.

    Chaque connecteur DOIT définir :
    - CONNECTOR_NAME  : identifiant unique ("hubspot", "pipedrive", etc.)
    - CONNECTOR_CATEGORY : catégorie ("crm", "email", "finance", etc.)
    - DATA_TYPES : types de données que ce connecteur produit
    """

    CONNECTOR_NAME: str = "base"
    CONNECTOR_CATEGORY: str = "base"
    DATA_TYPES: list[str] = []  # ex: ["deals", "contacts"] pour un CRM

    DEFAULT_DAYS_BACK: int = 90
    MAX_DAYS_BACK: int = 365
    MAX_RECORDS_PER_REQUEST: int = 100

    def __init__(self, company_id: str):
        self.company_id = company_id
        self.logger = logging.getLogger(
            f"kuria.connectors.{self.CONNECTOR_CATEGORY}.{self.CONNECTOR_NAME}"
        )
        self._authenticated = False
        self._metrics: Optional[ExtractionMetrics] = None

    async def __aenter__(self) -> "BaseConnector":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    @abstractmethod
    async def authenticate(self, **credentials: Any) -> None:
        ...

    @abstractmethod
    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    async def disconnect(self) -> None:
        self._authenticated = False
        self.logger.debug("Disconnected")

    def _require_auth(self) -> None:
        if not self._authenticated:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message="Not authenticated. Call authenticate() first.",
            )

    def _validate_days_back(self, days_back: int) -> int:
        if days_back < 1:
            return 1
        if days_back > self.MAX_DAYS_BACK:
            return self.MAX_DAYS_BACK
        return days_back

    def _start_metrics(self) -> ExtractionMetrics:
        self._metrics = ExtractionMetrics(connector_name=self.CONNECTOR_NAME)
        return self._metrics

    @property
    def last_metrics(self) -> Optional[ExtractionMetrics]:
        return self._metrics

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Metadata du connecteur pour le registry et le dashboard."""
        return {
            "name": cls.CONNECTOR_NAME,
            "category": cls.CONNECTOR_CATEGORY,
            "data_types": cls.DATA_TYPES,
        }

    def __repr__(self) -> str:
        auth = "authenticated" if self._authenticated else "not authenticated"
        return f"<{self.__class__.__name__} company={self.company_id} {auth}>"
