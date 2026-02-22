"""
SupabaseClient — Client Supabase + helpers CRUD typés.

UN client, initialisé UNE fois, partagé partout.

Tables Kuria :
  - companies          → CompanyProfile
  - agent_configs      → AgentConfigSet
  - runs               → RunResult
  - reports            → WeeklyReportData
  - frictions          → Friction
  - decisions          → DecisionLog
  - signals            → Signal
  - metrics_history    → MetricResult snapshots

Usage :
    from services.supabase import SupabaseClient

    db = SupabaseClient()
    company = await db.get_company("acme")
    await db.save_run(run_result)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from services.config import get_settings


class SupabaseClient:
    """
    Client Supabase avec helpers CRUD typés.

    Encapsule la librairie supabase-py.
    Expose des méthodes spécifiques par table.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._url = settings.supabase_url
        self._key = settings.supabase_service_key
        self._client: Any | None = None

        if settings.has_supabase:
            self._init_client()

    def _init_client(self) -> None:
        """Initialise le client Supabase."""
        try:
            from supabase import create_client, Client
            self._client: Client = create_client(self._url, self._key)
        except ImportError:
            self._client = None
        except Exception:
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────────────────
    # COMPANIES
    # ──────────────────────────────────────────────────────

    async def get_company(self, company_id: str) -> dict[str, Any] | None:
        """Récupère un profil d'entreprise."""
        return await self._get("companies", "id", company_id)

    async def save_company(self, data: dict[str, Any]) -> dict[str, Any]:
        """Crée ou met à jour un profil d'entreprise."""
        return await self._upsert("companies", data)

    async def list_companies(self, limit: int = 50) -> list[dict[str, Any]]:
        """Liste les entreprises."""
        return await self._list("companies", limit=limit)

    # ──────────────────────────────────────────────────────
    # AGENT CONFIGS
    # ──────────────────────────────────────────────────────

    async def get_config(self, company_id: str) -> dict[str, Any] | None:
        """Récupère la config des agents pour une entreprise."""
        return await self._get("agent_configs", "company_id", company_id)

    async def save_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde la config des agents."""
        return await self._upsert("agent_configs", data)

    # ──────────────────────────────────────────────────────
    # RUNS
    # ──────────────────────────────────────────────────────

    async def save_run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde un résultat d'exécution."""
        return await self._insert("runs", data)

    async def get_runs(
        self,
        company_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Récupère les dernières exécutions d'une entreprise."""
        return await self._list(
            "runs",
            filters={"company_id": company_id},
            limit=limit,
            order_by="created_at",
            ascending=False,
        )

    # ──────────────────────────────────────────────────────
    # REPORTS
    # ──────────────────────────────────────────────────────

    async def save_report(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde un rapport hebdomadaire."""
        return await self._insert("reports", data)

    async def get_latest_report(
        self, company_id: str
    ) -> dict[str, Any] | None:
        """Récupère le dernier rapport d'une entreprise."""
        results = await self._list(
            "reports",
            filters={"company_id": company_id},
            limit=1,
            order_by="created_at",
            ascending=False,
        )
        return results[0] if results else None

    async def get_reports(
        self,
        company_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Récupère les rapports d'une entreprise."""
        return await self._list(
            "reports",
            filters={"company_id": company_id},
            limit=limit,
            order_by="created_at",
            ascending=False,
        )

    # ──────────────────────────────────────────────────────
    # FRICTIONS
    # ──────────────────────────────────────────────────────

    async def save_friction(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde une friction détectée."""
        return await self._insert("frictions", data)

    async def save_frictions(
        self, frictions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sauvegarde plusieurs frictions."""
        results = []
        for f in frictions:
            r = await self._insert("frictions", f)
            results.append(r)
        return results

    async def get_frictions(
        self,
        company_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Récupère les frictions d'une entreprise."""
        filters: dict[str, Any] = {"company_id": company_id}
        if status:
            filters["status"] = status
        return await self._list("frictions", filters=filters, limit=limit)

    # ──────────────────────────────────────────────────────
    # SIGNALS
    # ──────────────────────────────────────────────────────

    async def save_signal(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde un signal inter-agent."""
        return await self._insert("signals", data)

    async def get_pending_signals(
        self, company_id: str
    ) -> list[dict[str, Any]]:
        """Récupère les signaux non traités."""
        return await self._list(
            "signals",
            filters={"company_id": company_id, "processed": False},
            limit=100,
        )

    # ──────────────────────────────────────────────────────
    # METRICS HISTORY
    # ──────────────────────────────────────────────────────

    async def save_metric(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sauvegarde un snapshot de métrique."""
        return await self._insert("metrics_history", data)

    async def get_metric_history(
        self,
        company_id: str,
        metric_name: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Récupère l'historique d'une métrique."""
        return await self._list(
            "metrics_history",
            filters={
                "company_id": company_id,
                "metric_name": metric_name,
            },
            limit=limit,
            order_by="created_at",
            ascending=False,
        )

    # ──────────────────────────────────────────────────────
    # CRUD GÉNÉRIQUE
    # ──────────────────────────────────────────────────────

    async def _get(
        self,
        table: str,
        key_column: str,
        key_value: str,
    ) -> dict[str, Any] | None:
        """SELECT single row."""
        if not self.is_connected:
            return None

        try:
            result = (
                self._client.table(table)
                .select("*")
                .eq(key_column, key_value)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]
            return None
        except Exception:
            return None

    async def _list(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        order_by: str | None = None,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """SELECT multiple rows."""
        if not self.is_connected:
            return []

        try:
            query = self._client.table(table).select("*")

            if filters:
                for col, val in filters.items():
                    query = query.eq(col, val)

            if order_by:
                query = query.order(order_by, desc=not ascending)

            query = query.limit(limit)
            result = query.execute()
            return result.data or []
        except Exception:
            return []

    async def _insert(
        self,
        table: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """INSERT single row."""
        if not self.is_connected:
            return data

        try:
            result = self._client.table(table).insert(data).execute()
            return result.data[0] if result.data else data
        except Exception:
            return data

    async def _upsert(
        self,
        table: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """UPSERT (insert or update)."""
        if not self.is_connected:
            return data

        try:
            result = self._client.table(table).upsert(data).execute()
            return result.data[0] if result.data else data
        except Exception:
            return data

    async def _delete(
        self,
        table: str,
        key_column: str,
        key_value: str,
    ) -> bool:
        """DELETE single row."""
        if not self.is_connected:
            return False

        try:
            self._client.table(table).delete().eq(key_column, key_value).execute()
            return True
        except Exception:
            return False
