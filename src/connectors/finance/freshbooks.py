"""
FreshBooks Connector — Comptabilité/facturation pour petites entreprises.

API REST v3. OAuth 2.0.
Rate limit : 100/min.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, safe_float


class FreshBooksConnector(BaseConnector):

    CONNECTOR_NAME = "freshbooks"
    CONNECTOR_CATEGORY = "finance"
    DATA_TYPES = ["accounts", "invoices", "bills", "summary"]

    BASE_URL = "https://api.freshbooks.com"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._account_id: Optional[str] = None

    async def authenticate(self, access_token: str, account_id: str, **kwargs) -> None:
        self._account_id = account_id
        self._client = httpx.AsyncClient(
            base_url=f"{self.BASE_URL}/accounting/account/{account_id}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/invoices/invoices?per_page=1")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid FreshBooks token.")
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
            return (await self._client.get("/invoices/invoices?per_page=1")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            raw_invoices = await self._fetch_all(
                "/invoices/invoices",
                "invoices",
                {"date_min": since},
            )
            invoices = [self._normalize_invoice(i) for i in raw_invoices]

            raw_expenses = await self._fetch_all(
                "/expenses/expenses",
                "expenses",
                {"date_min": since},
            )
            bills = [self._normalize_expense_as_bill(e) for e in raw_expenses]

            summary = self._calculate_summary(invoices, bills, days_back)
            metrics.complete(records=len(invoices) + len(bills))

            return {
                "accounts": [],
                "invoices": invoices,
                "bills": bills,
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _fetch_all(self, endpoint: str, key: str, params: Optional[dict] = None) -> list[dict]:
        assert self._client
        all_items = []
        page = 1
        p = dict(params or {})

        while True:
            p["page"] = page
            p["per_page"] = 100
            r = await self._client.get(endpoint, params=p)
            if r.status_code == 429:
                raise RateLimitError(self.CONNECTOR_NAME, 60)
            r.raise_for_status()
            data = r.json().get("response", {}).get("result", {})
            items = data.get(key, [])
            if not items:
                break
            all_items.extend(items)
            total_pages = data.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1

        return all_items

    def _normalize_invoice(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        amount = safe_float(raw.get("amount", {}).get("amount"), 0)
        outstanding = safe_float(raw.get("outstanding", {}).get("amount"), 0)
        due_date = normalize_date(raw.get("due_date"))
        issue_date = normalize_date(raw.get("create_date"))

        is_overdue = False
        days_overdue = 0
        if due_date and outstanding > 0:
            is_overdue = now > due_date
            if is_overdue:
                days_overdue = (now - due_date).days

        paid = raw.get("payment_status", "") == "paid"
        status = "paid" if paid else ("overdue" if is_overdue else "pending")

        return {
            "invoice_id": str(raw.get("invoiceid", raw.get("id", ""))),
            "customer_name": raw.get("organization", raw.get("fname", "")),
            "amount": amount,
            "amount_due": 0 if paid else outstanding,
            "issue_date": issue_date,
            "due_date": due_date,
            "status": status,
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "days_to_payment": None,
            "currency": raw.get("currency_code", "EUR"),
        }

    def _normalize_expense_as_bill(self, raw: dict) -> dict[str, Any]:
        amount = safe_float(raw.get("amount", {}).get("amount"), 0)
        return {
            "bill_id": str(raw.get("expenseid", raw.get("id", ""))),
            "vendor_name": raw.get("vendor", ""),
            "amount": amount,
            "amount_due": 0,
            "issue_date": normalize_date(raw.get("date")),
            "due_date": None,
            "status": "paid",
            "is_overdue": False,
            "category": raw.get("category", {}).get("category", "uncategorized"),
            "currency": raw.get("currency_code", "EUR"),
        }

    def _calculate_summary(self, invoices, bills, days_back) -> dict[str, Any]:
        months = max(1, days_back / 30)
        total_revenue = sum(i.get("amount", 0) for i in invoices)
        total_expenses = sum(b.get("amount", 0) for b in bills)
        overdue = [i for i in invoices if i.get("is_overdue")]

        return {
            "cash_position": 0,
            "total_receivable": sum(i.get("amount_due", 0) for i in invoices if i["status"] != "paid"),
            "total_payable": 0,
            "invoices_overdue_count": len(overdue),
            "invoices_overdue_value": round(sum(i.get("amount_due", 0) for i in overdue), 2),
            "avg_client_payment_days": 0,
            "avg_supplier_payment_days": 0,
            "monthly_burn_rate": round(total_expenses / months, 2),
            "monthly_revenue_avg": round(total_revenue / months, 2),
            "revenue_recurring_pct": 0,
            "runway_months": 99.0,
            "top_expenses": [],
        }
