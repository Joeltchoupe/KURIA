"""
Trello Connector — Kanban boards, cartes, listes.
API REST. API key + token.
Rate limit : 100/10s.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff


class TrelloConnector(BaseConnector):

    CONNECTOR_NAME = "trello"
    CONNECTOR_CATEGORY = "project"
    DATA_TYPES = ["tasks", "projects", "summary"]
    BASE_URL = "https://api.trello.com/1"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_params: dict[str, str] = {}

    async def authenticate(self, api_key: str, token: str, **kwargs) -> None:
        self._auth_params = {"key": api_key, "token": token}
        self._client = httpx.AsyncClient(base_url=self.BASE_URL, timeout=httpx.Timeout(30.0))
        try:
            r = await self._client.get("/members/me", params=self._auth_params)
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid credentials.")
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
            return (await self._client.get("/members/me", params=self._auth_params)).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()

        try:
            boards = await self._fetch_boards()
            all_tasks = []

            for board in boards[:10]:
                lists = await self._fetch_lists(board["id"])
                list_map = {l["id"]: l["name"] for l in lists}
                cards = await self._fetch_cards(board["id"])

                for card in cards:
                    all_tasks.append(self._normalize_card(card, list_map, board["name"]))

            summary = self._calculate_summary(all_tasks, boards)
            metrics.complete(records=len(all_tasks))

            return {
                "tasks": all_tasks,
                "projects": [{"project_id": b["id"], "name": b["name"]} for b in boards],
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        assert self._client
        p = {**self._auth_params, **(params or {})}
        r = await self._client.get(endpoint, params=p)
        if r.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, 10)
        r.raise_for_status()
        return r.json()

    async def _fetch_boards(self) -> list[dict]:
        return await self._api_get("/members/me/boards", {"filter": "open", "fields": "name,dateLastActivity"})

    async def _fetch_lists(self, board_id: str) -> list[dict]:
        return await self._api_get(f"/boards/{board_id}/lists", {"filter": "open", "fields": "name,pos"})

    async def _fetch_cards(self, board_id: str) -> list[dict]:
        return await self._api_get(f"/boards/{board_id}/cards", {"fields": "name,idList,due,dueComplete,idMembers,dateLastActivity,closed"})

    def _normalize_card(self, card: dict, list_map: dict, board_name: str) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        due = normalize_date(card.get("due"))
        completed = card.get("dueComplete", False) or card.get("closed", False)
        is_overdue = due is not None and not completed and now > due

        return {
            "task_id": card.get("id", ""),
            "name": card.get("name", ""),
            "assignee": None,
            "section": list_map.get(card.get("idList", ""), "Unknown"),
            "due_date": due,
            "completed": completed,
            "completed_at": None,
            "created_at": None,
            "modified_at": normalize_date(card.get("dateLastActivity")),
            "is_overdue": is_overdue,
            "days_overdue": (now - due).days if is_overdue and due else 0,
            "cycle_time_days": None,
        }

    def _calculate_summary(self, tasks, boards):
        total = len(tasks)
        overdue = [t for t in tasks if t.get("is_overdue")]
        completed = [t for t in tasks if t.get("completed")]
        return {
            "total_tasks": total, "completed_tasks": len(completed),
            "overdue_tasks": len(overdue), "unassigned_tasks": 0,
            "completion_rate": round(len(completed) / max(total, 1), 3),
            "avg_cycle_time_days": 0, "total_projects": len(boards),
            "person_load": [], "section_distribution": {},
}
