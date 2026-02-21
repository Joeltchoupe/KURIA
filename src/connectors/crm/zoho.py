"""
Zoho CRM Connector — Populaire chez les PME en Asie et en prix d'entrée.

API REST v5. OAuth 2.0.
Rate limit : 100 req/min (standard).
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, safe_float, truncate, retry_with_backoff


class ZohoConnector(BaseConnector):

    CONNECTOR_NAME = "zoho"
    CONNECTOR_CATEGORY = "crm"
    DATA_TYPES = ["deals", "contacts", "owners", "stages"]
    BASE_URL = "https://www.zohoapis.com/crm/v5"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, access_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/users?type=ActiveUsers")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid token.")
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
            return (await self._client.get("/users?type=ActiveUsers")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()
        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

        try:
            raw_deals = await self._fetch_records("Deals", since)
            raw_contacts = await self._fetch_records("Contacts", since)
            raw_users = await self._fetch_records("users", since, is_users=True)

            owners = {str(u.get("id", "")): u.get("full_name", "Unknown") for u in raw_users}
            deals = [self._normalize_deal(d, owners) for d in raw_deals]
            contacts = [self._normalize_contact(c) for c in raw_contacts]

            metrics.complete(records=len(deals) + len(contacts))
            return {
                "deals": deals, "contacts": contacts,
                "owners": owners, "stages": {},
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _fetch_records(self, module: str, since: datetime, is_users: bool = False) -> list[dict]:
        assert self._client
        all_records = []
        page = 1
        endpoint = f"/users?type=ActiveUsers" if is_users else f"/{module}"

        while True:
            params = {"page": page, "per_page": 200}
            if not is_users:
                params["modified_since"] = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            r = await self._client.get(endpoint, params=params)
            if r.status_code == 429:
                raise RateLimitError(self.CONNECTOR_NAME, 60)
            if r.status_code == 204:
                break
            r.raise_for_status()
            data = r.json()
