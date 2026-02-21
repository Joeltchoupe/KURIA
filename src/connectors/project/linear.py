"""Linear Connector — Issue tracking pour les équipes tech. GraphQL API. API key."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff


class LinearConnector(BaseConnector):
    CONNECTOR_NAME = "linear"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "projects", "summary"]
    BASE_URL = "https://api.linear.app/graphql"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, api_key: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(headers={"Authorization": api_key, "Content-Type": "application/json"}, timeout=httpx.Timeout(30.0))
        try:
            r = await self._client.post(self.BASE_URL, json={"query": "{ viewer { id name } }"})
            if r.status_code == 401: raise AuthenticationError(self.CONNECTOR_NAME, "Invalid key.")
            r.raise_for_status()
            self._authenticated = True
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(self.CONNECTOR_NAME, str(e), raw_error=e)

    async def disconnect(self) -> None:
        if self._client: await self._client.aclose()
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client: return False
        try:
            r = await self._client.post(self.BASE_URL, json={"query": "{ viewer { id } }"})
            return r.status_code == 200
        except Exception: return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()
        try:
            assert self._client
            query = """{ issues(first:200, orderBy:updatedAt) { nodes { id title state { name } assignee { name } dueDate createdAt updatedAt completedAt team { name } } } }"""
            r = await self._client.post(self.BASE_URL, json={"query": query})
            r.raise_for_status()
            issues = r.json().get("data", {}).get("issues", {}).get("nodes", [])
            now = datetime.now(tz=timezone.utc)
            tasks = []
            teams = set()
            for issue in issues:
                state = issue.get("state", {}).get("name", "") if issue.get("state") else ""
                completed = state.lower() in ("done", "closed", "cancelled", "terminé")
                due = normalize_date(issue.get("dueDate"))
                is_overdue = due is not None and not completed and now > due
                team = issue.get("team", {}).get("name") if issue.get("team") else None
                if team: teams.add(team)
                tasks.append({"task_id": issue.get("id",""), "name": issue.get("title",""), "assignee": issue.get("assignee",{}).get("name") if issue.get("assignee") else None, "section": state, "due_date": due, "completed": completed, "completed_at": normalize_date(issue.get("completedAt")), "created_at": normalize_date(issue.get("createdAt")), "modified_at": normalize_date(issue.get("updatedAt")), "is_overdue": is_overdue, "days_overdue": (now-due).days if is_overdue and due else 0, "cycle_time_days": None})

            completed_tasks = [t for t in tasks if t["completed"]]
            overdue_tasks = [t for t in tasks if t["is_overdue"]]
            summary = {"total_tasks": len(tasks), "completed_tasks": len(completed_tasks), "overdue_tasks": len(overdue_tasks), "unassigned_tasks": sum(1 for t in tasks if not t["assignee"] and not t["completed"]), "completion_rate": round(len(completed_tasks)/max(len(tasks),1),3), "avg_cycle_time_days": 0, "total_projects": len(teams), "person_load": [], "section_distribution": {}}
            metrics.complete(records=len(tasks))
            return {"tasks": tasks, "projects": [{"project_id": t, "name": t} for t in teams], "summary": summary, "extraction_metrics": metrics.model_dump()}
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)
