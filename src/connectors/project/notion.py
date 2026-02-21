"""
Notion Connector — Bases de données, pages, tâches.

API REST v2022-06-28. Bearer token (integration).
Rate limit : 3 req/s.

Notion est hybride : wiki + project management.
On extrait les databases qui ressemblent à des task boards.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff
import asyncio


class NotionConnector(BaseConnector):

    CONNECTOR_NAME = "notion"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "databases", "summary"]

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, integration_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {integration_token}",
                "Notion-Version": self.NOTION_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/users/me")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid Notion token.")
            r.raise_for_status()
            self._authenticated = True
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(self.CONNECTOR_NAME, str(e), raw_error=e)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client:
            return False
        try:
            return (await self._client.get("/users/me")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()

        try:
            databases = await self._fetch_databases()
            all_tasks = []

            for db in databases[:10]:
                pages = await self._query_database(db["id"])
                for page in pages:
                    task = self._normalize_page_as_task(page, db)
                    if task:
                        all_tasks.append(task)
                await asyncio.sleep(0.35)  # Rate limit : 3 req/s

            summary = self._calculate_summary(all_tasks, databases)
            metrics.complete(records=len(all_tasks))

            return {
                "tasks": all_tasks,
                "databases": [{"database_id": d["id"], "name": d.get("title", "")} for d in databases],
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_post(self, endpoint: str, body: dict) -> dict:
        assert self._client
        await asyncio.sleep(0.35)
        r = await self._client.post(endpoint, json=body)
        if r.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, 2)
        r.raise_for_status()
        return r.json()

    async def _fetch_databases(self) -> list[dict]:
        data = await self._api_post("/search", {"filter": {"value": "database", "property": "object"}})
        dbs = []
        for result in data.get("results", []):
            title_parts = result.get("title", [])
            title = title_parts[0].get("plain_text", "") if title_parts else "Untitled"
            dbs.append({"id": result["id"], "title": title, "properties": result.get("properties", {})})
        return dbs

    async def _query_database(self, database_id: str) -> list[dict]:
        all_pages = []
        start_cursor = None

        while True:
            body: dict[str, Any] = {"page_size": 100}
            if start_cursor:
                body["start_cursor"] = start_cursor

            data = await self._api_post(f"/databases/{database_id}/query", body)
            pages = data.get("results", [])
            all_pages.extend(pages)

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")

        return all_pages

    def _normalize_page_as_task(self, page: dict, db: dict) -> Optional[dict[str, Any]]:
        props = page.get("properties", {})
        now = datetime.now(tz=timezone.utc)

        # Extraire le titre
        name = ""
        for prop_name, prop_val in props.items():
            if prop_val.get("type") == "title":
                title_parts = prop_val.get("title", [])
                name = title_parts[0].get("plain_text", "") if title_parts else ""
                break

        if not name:
            return None

        # Status / Select
        status = None
        completed = False
        for prop_name, prop_val in props.items():
            if prop_val.get("type") == "status":
                status_obj = prop_val.get("status")
                if status_obj:
                    status = status_obj.get("name", "")
                    completed = status.lower() in ("done", "terminé", "complete", "completed", "fini")
            elif prop_val.get("type") == "select" and prop_name.lower() in ("status", "statut", "état"):
                select_obj = prop_val.get("select")
                if select_obj:
                    status = select_obj.get("name", "")

        # Assignee
        assignee = None
        for prop_name, prop_val in props.items():
            if prop_val.get("type") == "people":
                people = prop_val.get("people", [])
                if people:
                    assignee = people[0].get("name")

        # Due date
        due_date = None
        for prop_name, prop_val in props.items():
            if prop_val.get("type") == "date":
                date_obj = prop_val.get("date")
                if date_obj:
                    due_date = normalize_date(date_obj.get("start"))

        created = normalize_date(page.get("created_time"))
        modified = normalize_date(page.get("last_edited_time"))

        is_overdue = False
        days_overdue = 0
        if due_date and not completed:
            is_overdue = now > due_date
            if is_overdue:
                days_overdue = (now - due_date).days

        return {
            "task_id": page.get("id", ""),
            "name": name,
            "assignee": assignee,
            "section": status,
            "due_date": due_date,
            "completed": completed,
            "completed_at": modified if completed else None,
            "created_at": created,
            "modified_at": modified,
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "cycle_time_days": None,
        }

    def _calculate_summary(self, tasks, databases):
        total = len(tasks)
        completed = [t for t in tasks if t.get("completed")]
        overdue = [t for t in tasks if t.get("is_overdue")]
        unassigned = [t for t in tasks if not t.get("assignee") and not t.get("completed")]

        person_load: dict[str, int] = defaultdict(int)
        for t in tasks:
            if not t.get("completed") and t.get("assignee"):
                person_load[t["assignee"]] += 1

        return {
            "total_tasks": total,
            "completed_tasks": len(completed),
            "overdue_tasks": len(overdue),
            "unassigned_tasks": len(unassigned),
            "completion_rate": round(len(completed) / max(total, 1), 3),
            "avg_cycle_time_days": 0,
            "total_projects": len(databases),
            "person_load": [{"name": n, "open_tasks": c} for n, c in sorted(person_load.items(), key=lambda x: x[1], reverse=True)[:10]],
            "section_distribution": {},
  }
