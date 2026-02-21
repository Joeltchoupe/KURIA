"""Monday.com Connector — GraphQL API v2. API key header. Rate limit : 5000 complexity/min."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff


class MondayConnector(BaseConnector):
    CONNECTOR_NAME = "monday"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "projects", "summary"]
    BASE_URL = "https://api.monday.com/v2"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, api_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": api_token, "Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.post("", json={"query": "{ me { id name } }"})
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid token.")
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
            r = await self._client.post("", json={"query": "{ me { id } }"})
            return r.status_code == 200
        except Exception: return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()
        try:
            query = """{ boards(limit:20) { id name items_page(limit:200) { items { id name state column_values { id text } } } } }"""
            assert self._client
            r = await self._client.post("", json={"query": query})
            r.raise_for_status()
            data = r.json().get("data", {})
            boards = data.get("boards", [])

            all_tasks = []
            projects = []
            for board in boards:
                projects.append({"project_id": board["id"], "name": board["name"]})
                items = board.get("items_page", {}).get("items", [])
                for item in items:
                    all_tasks.append(self._normalize_item(item, board["name"]))

            now = datetime.now(tz=timezone.utc)
            total = len(all_tasks)
            completed = [t for t in all_tasks if t.get("completed")]
            overdue = [t for t in all_tasks if t.get("is_overdue")]
            summary = {"total_tasks": total, "completed_tasks": len(completed), "overdue_tasks": len(overdue), "unassigned_tasks": 0, "completion_rate": round(len(completed)/max(total,1),3), "avg_cycle_time_days": 0, "total_projects": len(projects), "person_load": [], "section_distribution": {}}
            metrics.complete(records=total)
            return {"tasks": all_tasks, "projects": projects, "summary": summary, "extraction_metrics": metrics.model_dump()}
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    def _normalize_item(self, item: dict, board_name: str) -> dict[str, Any]:
        completed = item.get("state", "") == "done" if item.get("state") else False
        cols = {c["id"]: c.get("text", "") for c in item.get("column_values", [])}
        due = None
        assignee = None
        status = None
        for col_id, val in cols.items():
            if "date" in col_id.lower() and val: due = normalize_date(val)
            if "person" in col_id.lower() and val: assignee = val
            if "status" in col_id.lower() and val: status = val

        now = datetime.now(tz=timezone.utc)
        is_overdue = due is not None and not completed and now > due
        return {"task_id": item.get("id",""), "name": item.get("name",""), "assignee": assignee, "section": status or board_name, "due_date": due, "completed": completed, "completed_at": None, "created_at": None, "modified_at": None, "is_overdue": is_overdue, "days_overdue": (now-due).days if is_overdue and due else 0, "cycle_time_days": None}
