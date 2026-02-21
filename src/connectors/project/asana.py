"""
Asana Connector — Gestion de projet, tâches, workflows.

API REST v1. Bearer token (PAT ou OAuth).
Rate limit : 1500/min.

Données critiques pour l'agent Process Clarity :
- Tasks (statut, assigné, dates, retard)
- Projects (workflows, sections = stages)
- Workspaces (structure organisationnelle)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, calculate_business_days


class AsanaConnector(BaseConnector):

    CONNECTOR_NAME = "asana"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "projects", "sections", "summary"]

    BASE_URL = "https://app.asana.com/api/1.0"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._workspace_id: Optional[str] = None

    async def authenticate(self, access_token: str, workspace_id: Optional[str] = None, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/users/me")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid token.")
            r.raise_for_status()
            user_data = r.json().get("data", {})

            if workspace_id:
                self._workspace_id = workspace_id
            else:
                workspaces = user_data.get("workspaces", [])
                if workspaces:
                    self._workspace_id = workspaces[0].get("gid")

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
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            # Projects
            raw_projects = await self._fetch_projects()
            projects = [self._normalize_project(p) for p in raw_projects]

            # Tasks
            all_tasks = []
            for project in raw_projects[:20]:  # Cap pour éviter les timeouts
                project_tasks = await self._fetch_project_tasks(project["gid"], since)
                all_tasks.extend(project_tasks)

            tasks = [self._normalize_task(t) for t in all_tasks]

            summary = self._calculate_summary(tasks, projects)
            metrics.complete(records=len(tasks) + len(projects))

            return {
                "tasks": tasks,
                "projects": projects,
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "30"))
            raise RateLimitError(self.CONNECTOR_NAME, retry_after)
        r.raise_for_status()
        return r.json()

    async def _fetch_projects(self) -> list[dict]:
        data = await self._api_get(
            "/projects",
            params={"workspace": self._workspace_id, "opt_fields": "name,created_at,modified_at,current_status,team.name"},
        )
        return data.get("data", [])

    async def _fetch_project_tasks(self, project_gid: str, since: str) -> list[dict]:
        all_tasks = []
        offset = None

        while True:
            params = {
                "project": project_gid,
                "opt_fields": "name,assignee.name,due_on,completed,completed_at,created_at,modified_at,memberships.section.name",
                "modified_since": f"{since}T00:00:00.000Z",
                "limit": 100,
            }
            if offset:
                params["offset"] = offset

            data = await self._api_get("/tasks", params)
            tasks = data.get("data", [])
            all_tasks.extend(tasks)

            next_page = data.get("next_page")
            if not next_page:
                break
            offset = next_page.get("offset")

        return all_tasks

    def _normalize_project(self, raw: dict) -> dict[str, Any]:
        return {
            "project_id": raw.get("gid", ""),
            "name": raw.get("name", ""),
            "team": raw.get("team", {}).get("name") if raw.get("team") else None,
            "created_at": normalize_date(raw.get("created_at")),
            "modified_at": normalize_date(raw.get("modified_at")),
            "status": raw.get("current_status", {}).get("text") if raw.get("current_status") else None,
        }

    def _normalize_task(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        due_on = normalize_date(raw.get("due_on"))
        completed = raw.get("completed", False)
        completed_at = normalize_date(raw.get("completed_at"))
        created_at = normalize_date(raw.get("created_at"))

        is_overdue = False
        days_overdue = 0
        if due_on and not completed:
            is_overdue = now > due_on
            if is_overdue:
                days_overdue = (now - due_on).days

        # Section = stage dans le workflow
        section = None
        memberships = raw.get("memberships", [])
        if memberships:
            section_data = memberships[0].get("section", {})
            section = section_data.get("name") if section_data else None

        # Cycle time (si complété)
        cycle_time_days = None
        if completed and completed_at and created_at:
            cycle_time_days = calculate_business_days(created_at, completed_at)

        assignee = raw.get("assignee")
        assignee_name = assignee.get("name") if assignee else None

        return {
            "task_id": raw.get("gid", ""),
            "name": raw.get("name", ""),
            "assignee": assignee_name,
            "section": section,
            "due_date": due_on,
            "completed": completed,
            "completed_at": completed_at,
            "created_at": created_at,
            "modified_at": normalize_date(raw.get("modified_at")),
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "cycle_time_days": cycle_time_days,
        }

    def _calculate_summary(self, tasks: list[dict], projects: list[dict]) -> dict[str, Any]:
        total = len(tasks)
        completed = [t for t in tasks if t.get("completed")]
        overdue = [t for t in tasks if t.get("is_overdue")]
        unassigned = [t for t in tasks if not t.get("assignee") and not t.get("completed")]

        cycle_times = [t["cycle_time_days"] for t in tasks if t.get("cycle_time_days") is not None]
        avg_cycle = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        # Charge par personne
        person_load: dict[str, int] = defaultdict(int)
        for t in tasks:
            if not t.get("completed") and t.get("assignee"):
                person_load[t["assignee"]] += 1

        busiest = sorted(person_load.items(), key=lambda x: x[1], reverse=True)

        # Tâches par section (= distribution dans le workflow)
        section_counts: dict[str, int] = defaultdict(int)
        for t in tasks:
            if not t.get("completed") and t.get("section"):
                section_counts[t["section"]] += 1

        return {
            "total_tasks": total,
            "completed_tasks": len(completed),
            "overdue_tasks": len(overdue),
            "unassigned_tasks": len(unassigned),
            "completion_rate": round(len(completed) / max(total, 1), 3),
            "avg_cycle_time_days": round(avg_cycle, 1),
            "total_projects": len(projects),
            "person_load": [{"name": n, "open_tasks": c} for n, c in busiest[:10]],
            "section_distribution": dict(section_counts),
}
