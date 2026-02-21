"""
Salesforce Connector — Le CRM enterprise.

API REST v57.0. OAuth 2.0.
Rate limit : 100,000 calls/24h (enterprise) ou 15,000 (pro).
Utilise SOQL pour les requêtes.

Complexité plus élevée que HubSpot/Pipedrive :
- Modèle de données plus riche (Opportunity vs Deal, Account vs Contact)
- Custom fields omniprésents
- Pagination via nextRecordsUrl
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
from connectors.utils import normalize_date, safe_float, truncate, retry_with_backoff


class SalesforceConnector(BaseConnector):

    CONNECTOR_NAME = "salesforce"
    CONNECTOR_CATEGORY = "crm"
    DATA_TYPES = ["deals", "contacts", "owners", "stages"]
    API_VERSION = "v57.0"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._instance_url: Optional[str] = None

    async def authenticate(self, access_token: str, instance_url: str, **kwargs) -> None:
        self._instance_url = instance_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=f"{self._instance_url}/services/data/{self.API_VERSION}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/sobjects")
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
            return (await self._client.get("/limits")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            # Opportunities (= Deals)
            opps = await self._soql_query(
                f"SELECT Id, Name, StageName, Amount, CloseDate, CreatedDate, "
                f"LastModifiedDate, OwnerId, IsClosed, IsWon, Probability, "
                f"NextStep, LeadSource, ForecastCategory "
                f"FROM Opportunity WHERE LastModifiedDate >= {since}"
            )

            # Contacts
            contacts = await self._soql_query(
                f"SELECT Id, Email, FirstName, LastName, Account.Name, Title, "
                f"LeadSource, CreatedDate, LastActivityDate, OwnerId "
                f"FROM Contact WHERE CreatedDate >= {since}"
            )

            # Users (Owners)
            users = await self._soql_query("SELECT Id, Name FROM User WHERE IsActive = true")
            owners = {u["Id"]: u["Name"] for u in users}

            deals = [self._normalize_deal(o, owners) for o in opps]
            normalized_contacts = [self._normalize_contact(c) for c in contacts]

            metrics.complete(records=len(deals) + len(normalized_contacts))

            return {
                "deals": deals,
                "contacts": normalized_contacts,
                "owners": owners,
                "stages": {},
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, f"Extraction failed: {e}", raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _soql_query(self, query: str) -> list[dict]:
        assert self._client
        all_records = []
        url = f"/query?q={query}"
        while url:
            r = await self._client.get(url)
            if r.status_code == 429:
                raise RateLimitError(self.CONNECTOR_NAME, 60)
            r.raise_for_status()
            data = r.json()
            all_records.extend(data.get("records", []))
            url = data.get("nextRecordsUrl")
        return all_records

    def _normalize_deal(self, raw: dict, owners: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        last_mod = normalize_date(raw.get("LastModifiedDate"))
        days_since = (now - last_mod).days if last_mod else 0

        return {
            "deal_id": raw.get("Id", ""),
            "name": truncate(raw.get("Name", ""), 256),
            "stage": raw.get("StageName", ""),
            "pipeline": raw.get("ForecastCategory", ""),
            "amount": safe_float(raw.get("Amount"), 0),
            "close_date": normalize_date(raw.get("CloseDate")),
            "create_date": normalize_date(raw.get("CreatedDate")),
            "owner_id": raw.get("OwnerId", ""),
            "owner_name": owners.get(raw.get("OwnerId", ""), "Unknown"),
            "is_closed": raw.get("IsClosed", False),
            "is_won": raw.get("IsWon", False),
            "days_in_current_stage": days_since,
            "days_since_last_activity": days_since,
            "last_activity_date": last_mod,
            "last_activity_type": None,
            "activity_count_30d": 0,
            "has_next_step": bool(raw.get("NextStep")),
            "contact_ids": [],
            "source": raw.get("LeadSource"),
            "raw_properties": raw,
        }

    def _normalize_contact(self, raw: dict) -> dict[str, Any]:
        account = raw.get("Account") or {}
        return {
            "contact_id": raw.get("Id", ""),
            "email": raw.get("Email"),
            "first_name": raw.get("FirstName"),
            "last_name": raw.get("LastName"),
            "company": account.get("Name"),
            "job_title": raw.get("Title"),
            "lifecycle_stage": None,
            "lead_status": None,
            "source": raw.get("LeadSource"),
            "source_detail": None,
            "create_date": normalize_date(raw.get("CreatedDate")),
            "last_activity_date": normalize_date(raw.get("LastActivityDate")),
            "owner_id": raw.get("OwnerId"),
            "associated_deal_ids": [],
}
