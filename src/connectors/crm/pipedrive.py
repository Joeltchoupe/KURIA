"""
Pipedrive Connector — CRM très populaire chez les PME européennes.

API REST v1. Pagination cursor-based.
Rate limit : 100 req/10s (similaire HubSpot).

Output normalisé IDENTIQUE à HubSpot :
même NormalizedDeal, même NormalizedContact.
Les agents ne font AUCUNE différence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from connectors.base import (
    AuthenticationError,
    BaseConnector,
    ConnectorError,
    RateLimitError,
)
from connectors.utils import (
    normalize_date,
    retry_with_backoff,
    safe_float,
    safe_int,
    truncate,
)


class PipedriveConnector(BaseConnector):

    CONNECTOR_NAME = "pipedrive"
    CONNECTOR_CATEGORY = "crm"
    DATA_TYPES = ["deals", "contacts", "owners", "stages"]
    MAX_RECORDS_PER_REQUEST = 100

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._api_token: Optional[str] = None
        self._base_url: str = ""
        self._users_cache: dict[str, str] = {}
        self._stages_cache: dict[str, dict[str, Any]] = {}

    async def authenticate(self, api_token: str, domain: str = "", **kwargs) -> None:
        """Pipedrive utilise un API token + company domain.

        Args:
            api_token: Le API token Pipedrive
            domain: Le sous-domaine company (ex: "acme" → acme.pipedrive.com)
        """
        self._api_token = api_token
        self._base_url = f"https://{domain}.pipedrive.com/api/v1" if domain else "https://api.pipedrive.com/api/v1"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        try:
            response = await self._client.get(
                "/users/me",
                params={"api_token": api_token},
            )
            if response.status_code == 401:
                raise AuthenticationError(
                    connector_name=self.CONNECTOR_NAME,
                    message="Invalid Pipedrive API token.",
                )
            response.raise_for_status()
            self._authenticated = True
            self.logger.info("Authenticated successfully")
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Auth failed: {e.response.status_code}",
                raw_error=e,
            )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client:
            return False
        try:
            r = await self._client.get(
                "/users/me",
                params={"api_token": self._api_token},
            )
            return r.status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        total_records = 0
        total_errors = 0

        try:
            # Users (owners)
            self._users_cache = await self._fetch_users()

            # Stages
            self._stages_cache = await self._fetch_stages()

            # Deals
            since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
            raw_deals = await self._fetch_deals(since)
            deals = []
            for raw in raw_deals:
                try:
                    deals.append(self._normalize_deal(raw))
                except Exception as e:
                    total_errors += 1
                    self.logger.warning(f"Failed to normalize deal: {e}")
            total_records += len(deals)

            # Persons (contacts)
            raw_persons = await self._fetch_persons(since)
            contacts = []
            for raw in raw_persons:
                try:
                    contacts.append(self._normalize_contact(raw))
                except Exception as e:
                    total_errors += 1
                    self.logger.warning(f"Failed to normalize person: {e}")
            total_records += len(contacts)

            metrics.complete(records=total_records, errors=total_errors)

            return {
                "deals": [d.to_dict() if hasattr(d, 'to_dict') else d for d in deals],
                "contacts": [c.to_dict() if hasattr(c, 'to_dict') else c for c in contacts],
                "owners": self._users_cache,
                "stages": self._stages_cache,
                "extraction_metrics": metrics.model_dump(),
            }

        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Extraction failed: {e}",
                raw_error=e,
            )

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        self._require_auth()
        assert self._client
        p = dict(params or {})
        p["api_token"] = self._api_token

        response = await self._client.get(endpoint, params=p)
        if response.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, retry_after_seconds=10)
        if response.status_code == 401:
            raise AuthenticationError(self.CONNECTOR_NAME, "Token expired.")
        response.raise_for_status()
        return response.json()

    async def _fetch_users(self) -> dict[str, str]:
        data = await self._api_get("/users")
        users = {}
        for u in data.get("data", []) or []:
            users[str(u.get("id", ""))] = u.get("name", "Unknown")
        return users

    async def _fetch_stages(self) -> dict[str, dict[str, Any]]:
        data = await self._api_get("/stages")
        stages = {}
        for s in data.get("data", []) or []:
            stages[str(s.get("id", ""))] = {
                "label": s.get("name", ""),
                "display_order": s.get("order_nr", 0),
                "pipeline_id": str(s.get("pipeline_id", "")),
                "pipeline_label": s.get("pipeline_name", ""),
                "is_closed": s.get("deal_probability", 100) in (0, 100),
            }
        return stages

    async def _fetch_deals(self, since: datetime) -> list[dict]:
        all_deals = []
        start = 0
        while True:
            data = await self._api_get("/deals", params={
                "start": start,
                "limit": self.MAX_RECORDS_PER_REQUEST,
                "sort": "update_time DESC",
            })
            items = data.get("data") or []
            if not items:
                break

            for d in items:
                update_time = normalize_date(d.get("update_time"))
                if update_time and update_time >= since:
                    all_deals.append(d)

            pagination = data.get("additional_data", {}).get("pagination", {})
            if not pagination.get("more_items_in_collection"):
                break
            start = pagination.get("next_start", start + len(items))

        return all_deals

    async def _fetch_persons(self, since: datetime) -> list[dict]:
        all_persons = []
        start = 0
        while True:
            data = await self._api_get("/persons", params={
                "start": start,
                "limit": self.MAX_RECORDS_PER_REQUEST,
                "sort": "update_time DESC",
            })
            items = data.get("data") or []
            if not items:
                break

            for p in items:
                add_time = normalize_date(p.get("add_time"))
                if add_time and add_time >= since:
                    all_persons.append(p)

            pagination = data.get("additional_data", {}).get("pagination", {})
            if not pagination.get("more_items_in_collection"):
                break
            start = pagination.get("next_start", start + len(items))

        return all_persons

    def _normalize_deal(self, raw: dict) -> dict[str, Any]:
        """Normalise vers le même format que HubSpot. Les agents ne voient pas la différence."""
        now = datetime.now(tz=timezone.utc)
        stage_id = str(raw.get("stage_id", ""))
        stage_info = self._stages_cache.get(stage_id, {})
        owner_id = str(raw.get("user_id", ""))
        update_time = normalize_date(raw.get("update_time"))
        add_time = normalize_date(raw.get("add_time"))

        days_since_activity = 0
        if update_time:
            days_since_activity = max(0, (now - update_time).days)

        status = raw.get("status", "open")
        is_closed = status in ("won", "lost", "deleted")
        is_won = status == "won"

        return {
            "deal_id": str(raw.get("id", "")),
            "name": truncate(raw.get("title", "Untitled"), 256),
            "stage": stage_info.get("label", stage_id),
            "pipeline": stage_info.get("pipeline_label", ""),
            "amount": safe_float(raw.get("value"), default=0.0),
            "close_date": normalize_date(raw.get("close_time")) if is_closed else normalize_date(raw.get("expected_close_date")),
            "create_date": add_time,
            "owner_id": owner_id,
            "owner_name": self._users_cache.get(owner_id, "Unknown"),
            "is_closed": is_closed,
            "is_won": is_won,
            "days_in_current_stage": safe_int(raw.get("stage_order_nr"), default=0),
            "days_since_last_activity": days_since_activity,
            "last_activity_date": update_time,
            "last_activity_type": None,
            "activity_count_30d": safe_int(raw.get("activities_count"), default=0),
            "has_next_step": bool(raw.get("next_activity_id")),
            "contact_ids": [str(pid) for pid in (raw.get("person_id") or []) if pid] if isinstance(raw.get("person_id"), list) else ([str(raw["person_id"])] if raw.get("person_id") else []),
            "source": raw.get("channel"),
            "raw_properties": raw,
        }

    def _normalize_contact(self, raw: dict) -> dict[str, Any]:
        """Normalise un person Pipedrive vers le format contact standard."""
        emails = raw.get("email", [])
        email = emails[0].get("value", "") if isinstance(emails, list) and emails else ""
        phones = raw.get("phone", [])

        return {
            "contact_id": str(raw.get("id", "")),
            "email": email or None,
            "first_name": raw.get("first_name"),
            "last_name": raw.get("last_name"),
            "company": raw.get("org_name"),
            "job_title": None,
            "lifecycle_stage": None,
            "lead_status": None,
            "source": None,
            "source_detail": None,
            "create_date": normalize_date(raw.get("add_time")),
            "last_activity_date": normalize_date(raw.get("last_activity_date")),
            "owner_id": str(raw.get("owner_id", "")),
            "associated_deal_ids": [],
}
