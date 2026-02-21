"""
ClickUp Connector — Project management tout-en-un.
API REST v2. Bearer token.
Rate limit : 100/min.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff


class ClickUpConnector(BaseConnector):

    CONNECTOR_NAME = "clickup"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "projects", "summary"]
    BASE_URL = "https://api.clickup.com/api/v2"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._team_id: Optional[str] = None

    async def authenticate(self, access_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": access_token},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/team")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid token.")
            r.raise_for_status()
            teams = r.json().get("teams", [])
            if teams:
                self._team_id = teams[0].get("id")
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
            return (await self._client.get("/team")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()

        try:
            spaces = await self._fetch_spaces()
            all_tasks = []
            all_projects = []

            for space in spaces[:10]:
                folders = await self._fetch_folders(space["id"])
                for folder in folders:
                    lists = await self._fetch_lists(folder["id"])
                    for lst in lists:
                        all_projects.append({"project_id": lst["id"], "name": lst.get("name", "")})
                        tasks = await self._fetch_tasks(lst["id"])
                        for t in tasks:
                            all_tasks.append(self._normalize_task(t, lst.get("name", "")))

            summary = self._calculate_summary(all_tasks, all_projects)
            metrics.complete(records=len(all_tasks))
            return {"tasks": all_tasks, "projects": all_projects, "summary": summary, "extraction_metrics": metrics.model_dump()}
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        if r.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, 60)
        r.raise_for_status()
        return r.json()

    async def _fetch_spaces(self) -> list[dict]:
        data = await self._api_get(f"/team/{self._team_id}/space")
        return data.get("spaces", [])

    async def _fetch_folders(self, space_id: str) -> list[dict]:
        data = await self._api_get(f"/space/{space_id}/folder")
        return data.get("folders", [])

    async def _fetch_lists(self, folder_id: str) -> list[dict]:
        data = await self._api_get(f"/folder/{folder_id}/list")
        return data.get("lists", [])

    async def _fetch_tasks(self, list_id: str) -> list[dict]:
        data = await self._api_get(f"/list/{list_id}/task", {"include_closed": "true"})
        return data.get("tasks", [])

    def _normalize_task(self, raw: dict, list_name: str) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        due = normalize_date(raw.get("due_date"))
        status = raw.get("status", {})
        status_name = status.get("status", "") if isinstance(status, dict) else ""
        completed = status_name.lower() in ("closed", "done", "complete", "terminé")

        assignees = raw.get("assignees", [])
        assignee = assignees[0].get("username", assignees[0].get("email")) if assignees else None

        is_overdue = due is not None and not completed and now > due

        return {
            "task_id": raw.get("id", ""),
            "name": raw.get("name", ""),
            "assignee": assignee,
            "section": status_name or list_name,
            "due_date": due,
            "completed": completed,
            "completed_at": normalize_date(raw.get("date_closed")),
            "created_at": normalize_date(raw.get("date_created")),
            "modified_at": normalize_date(raw.get("date_updated")),
            "is_overdue": is_overdue,
            "days_overdue": (now - due).days if is_overdue and due else 0,
            "cycle_time_days": None,
        }

    def _calculate_summary(self, tasks, projects):
        total = len(tasks)
        completed = [t for t in tasks if t.get("completed")]
        overdue = [t for t in tasks if t.get("is_overdue")]
        unassigned = [t for t in tasks if not t.get("assignee") and not t.get("completed")]
        person_load: dict[str, int] = defaultdict(int)
        for t in tasks:
            if not t.get("completed") and t.get("assignee"):
                person_load[t["assignee"]] += 1
        return {
            "total_tasks": total, "completed_tasks": len(completed),
            "overdue_tasks": len(overdue), "unassigned_tasks": len(unassigned),
            "completion_rate": round(len(completed) / max(total, 1), 3),
            "avg_cycle_time_days": 0, "total_projects": len(projects),
            "person_load": [{"name": n, "open_tasks": c} for n, c in sorted(person_load.items(), key=lambda x: x[1], reverse=True)[:10]],
            "section_distribution": {},
      }
