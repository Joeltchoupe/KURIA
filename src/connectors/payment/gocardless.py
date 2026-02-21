"""
GoCardless Connector — Prélèvements SEPA, populaire en Europe.

API REST v2. Bearer token.
Rate limit : 1000/min.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, safe_float


class GoCardlessConnector(BaseConnector):

    CONNECTOR_NAME = "gocardless"
    CONNECTOR_CATEGORY = "payment"
    DATA_TYPES = ["payments", "subscriptions", "summary"]

    BASE_URL = "https://api.gocardless.com"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, access_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "GoCardless-Version": "2015-07-06",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/creditors")
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
            return (await self._client.get("/creditors")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            raw_payments = await self._paginate("/payments", {"created_at[gte]": since})
            payments = [self._normalize_payment(p) for p in raw_payments]

            raw_subs = await self._paginate("/subscriptions", {"status": "active"})
            subscriptions = [self._normalize_subscription(s) for s in raw_subs]

            invoices = [p for p in payments if p.get("status") in ("confirmed", "paid_out")]
            mrr = sum(s.get("mrr", 0) for s in subscriptions)

            months = max(1, days_back / 30)
            total_revenue = sum(p.get("amount", 0) for p in invoices)

            summary = {
                "cash_position": 0,
                "total_receivable": sum(p.get("amount", 0) for p in payments if p["status"] == "pending_submission"),
                "total_payable": 0,
                "invoices_overdue_count": sum(1 for p in payments if p["status"] == "failed"),
                "invoices_overdue_value": sum(p.get("amount", 0) for p in payments if p["status"] == "failed"),
                "avg_client_payment_days": 0,
                "avg_supplier_payment_days": 0,
                "monthly_burn_rate": 0,
                "monthly_revenue_avg": round(total_revenue / months, 2),
                "revenue_recurring_pct": round(mrr * 12 / max(total_revenue, 1), 3) if total_revenue else 0,
                "runway_months": 99.0,
                "top_expenses": [],
                "mrr": round(mrr, 2),
                "active_subscriptions": len(subscriptions),
            }

            metrics.complete(records=len(payments) + len(subscriptions))
            return {
                "accounts": [],
                "invoices": [{
                    "invoice_id": p["payment_id"], "customer_name": p.get("customer_id", ""),
                    "amount": p["amount"], "amount_due": 0 if p["status"] in ("confirmed", "paid_out") else p["amount"],
                    "issue_date": p["date"], "due_date": p["date"],
                    "status": "paid" if p["status"] in ("confirmed", "paid_out") else "pending",
                    "is_overdue": p["status"] == "failed", "days_overdue": 0,
                    "days_to_payment": None, "currency": p["currency"],
                } for p in payments],
                "bills": [],
                "subscriptions": subscriptions,
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _paginate(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        assert self._client
        all_items = []
        p = dict(params or {})
        p["limit"] = 100

        while True:
            r = await self._client.get(endpoint, params=p)
            if r.status_code == 429:
                raise RateLimitError(self.CONNECTOR_NAME, 60)
            r.raise_for_status()
            data = r.json()
            key = [k for k in data if k not in ("meta",)][0] if data else ""
            items = data.get(key, [])
            if not items:
                break
            all_items.extend(items)

            cursors = data.get("meta", {}).get("cursors", {})
            after = cursors.get("after")
            if not after:
                break
            p["after"] = after

        return all_items

    def _normalize_payment(self, raw: dict) -> dict[str, Any]:
        return {
            "payment_id": raw.get("id", ""),
            "amount": (raw.get("amount", 0) or 0) / 100,
            "currency": raw.get("currency", "EUR").upper(),
            "status": raw.get("status", ""),
            "date": normalize_date(raw.get("created_at")),
            "customer_id": raw.get("links", {}).get("mandate", ""),
            "description": raw.get("description"),
        }

    def _normalize_subscription(self, raw: dict) -> dict[str, Any]:
        amount = (raw.get("amount", 0) or 0) / 100
        interval = raw.get("interval_unit", "monthly")
        mrr = amount if interval == "monthly" else (amount / 12 if interval == "yearly" else amount * 4.33)

        return {
            "subscription_id": raw.get("id", ""),
            "status": raw.get("status", ""),
            "mrr": round(mrr, 2),
            "amount": amount,
            "interval": interval,
            "created": normalize_date(raw.get("created_at")),
  }
