"""
Sage Connector — ERP/Comptabilité historique, très répandu chez les PME françaises.

Sage Business Cloud API. OAuth 2.0.
Rate limit : variable selon le plan (safe : 30/min).

Sage a un modèle de données plus rigide que les SaaS modernes.
Les noms d'endpoints changent selon la version (Sage 50, Sage 100, Sage Business Cloud).
Ce connecteur cible Sage Business Cloud Accounting.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, safe_float


class SageConnector(BaseConnector):

    CONNECTOR_NAME = "sage"
    CONNECTOR_CATEGORY = "finance"
    DATA_TYPES = ["accounts", "invoices", "bills", "summary"]

    BASE_URL = "https://api.accounting.sage.com/v3.1"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, access_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/business")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid Sage token.")
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
            return (await self._client.get("/business")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            # Bank accounts
            raw_accounts = await self._paginate("/bank_accounts")
            accounts = [
                {
                    "account_id": a.get("id", ""),
                    "name": a.get("bank_account_details", {}).get("account_name", a.get("displayed_as", "")),
                    "account_type": "BANK",
                    "balance": safe_float(a.get("balance"), 0),
                    "currency": a.get("currency", {}).get("id", "EUR"),
                }
                for a in raw_accounts
            ]

            # Sales invoices
            raw_invoices = await self._paginate(
                "/sales_invoices",
                params={"from_date": since, "attributes": "items"},
            )
            invoices = [self._normalize_invoice(i) for i in raw_invoices]

            # Purchase invoices
            raw_bills = await self._paginate(
                "/purchase_invoices",
                params={"from_date": since, "attributes": "items"},
            )
            bills = [self._normalize_bill(b) for b in raw_bills]

            summary = self._calculate_summary(accounts, invoices, bills, days_back)
            metrics.complete(records=len(accounts) + len(invoices) + len(bills))

            return {
                "accounts": accounts,
                "invoices": invoices,
                "bills": bills,
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
        page = 1
        p = dict(params or {})

        while True:
            p["page"] = page
            p["items_per_page"] = 100
            r = await self._client.get(endpoint, params=p)
            if r.status_code == 429:
                raise RateLimitError(self.CONNECTOR_NAME, 60)
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Token expired.")
            r.raise_for_status()
            data = r.json()

            # Sage wraps dans $items
            items = data.get("$items", [])
            if not items:
                break
            all_items.extend(items)

            total = data.get("$total", 0)
            if len(all_items) >= total:
                break
            page += 1

        return all_items

    def _normalize_invoice(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        amount = safe_float(raw.get("total_amount"), 0)
        outstanding = safe_float(raw.get("outstanding_amount"), 0)
        due_date = normalize_date(raw.get("due_date"))
        issue_date = normalize_date(raw.get("date"))

        paid = raw.get("status", {}).get("id", "") == "PAID"
        is_overdue = False
        days_overdue = 0
        if due_date and not paid and outstanding > 0:
            is_overdue = now > due_date
            if is_overdue:
                days_overdue = (now - due_date).days

        status = "paid" if paid else ("overdue" if is_overdue else "pending")
        contact = raw.get("contact", {})

        return {
            "invoice_id": raw.get("id", ""),
            "customer_name": contact.get("displayed_as", ""),
            "amount": amount,
            "amount_due": 0 if paid else outstanding,
            "issue_date": issue_date,
            "due_date": due_date,
            "status": status,
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "days_to_payment": None,
            "currency": raw.get("currency", {}).get("id", "EUR"),
        }

    def _normalize_bill(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        amount = safe_float(raw.get("total_amount"), 0)
        outstanding = safe_float(raw.get("outstanding_amount"), 0)
        due_date = normalize_date(raw.get("due_date"))
        paid = raw.get("status", {}).get("id", "") == "PAID"
        is_overdue = not paid and due_date is not None and now > due_date and outstanding > 0

        contact = raw.get("contact", {})
        category = "uncategorized"
        items = raw.get("invoice_lines", [])
        if items:
            ledger = items[0].get("ledger_account", {})
            category = ledger.get("displayed_as", "uncategorized")

        return {
            "bill_id": raw.get("id", ""),
            "vendor_name": contact.get("displayed_as", ""),
            "amount": amount,
            "amount_due": 0 if paid else outstanding,
            "issue_date": normalize_date(raw.get("date")),
            "due_date": due_date,
            "status": "paid" if paid else ("overdue" if is_overdue else "pending"),
            "is_overdue": is_overdue,
            "category": category,
            "currency": raw.get("currency", {}).get("id", "EUR"),
        }

    def _calculate_summary(self, accounts, invoices, bills, days_back) -> dict[str, Any]:
        cash_position = sum(a.get("balance", 0) for a in accounts)
        months = max(1, days_back / 30)
        total_revenue = sum(i.get("amount", 0) for i in invoices)
        total_expenses = sum(b.get("amount", 0) for b in bills)
        overdue = [i for i in invoices if i.get("is_overdue")]

        payment_days = [i["days_to_payment"] for i in invoices if i.get("days_to_payment") and i["days_to_payment"] > 0]
        avg_client_days = sum(payment_days) / len(payment_days) if payment_days else 0

        monthly_burn = total_expenses / months
        monthly_revenue = total_revenue / months
        net = monthly_revenue - monthly_burn
        runway = 99.0
        if net < 0:
            runway = cash_position / abs(net) if net != 0 else 0

        expense_cats: dict[str, float] = {}
        for b in bills:
            cat = b.get("category", "uncategorized")
            expense_cats[cat] = expense_cats.get(cat, 0) + b.get("amount", 0)
        top_exp = sorted(expense_cats.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "cash_position": round(cash_position, 2),
            "total_receivable": round(sum(i.get("amount_due", 0) for i in invoices if i["status"] != "paid"), 2),
            "total_payable": round(sum(b.get("amount_due", 0) for b in bills if b["status"] != "paid"), 2),
            "invoices_overdue_count": len(overdue),
            "invoices_overdue_value": round(sum(i.get("amount_due", 0) for i in overdue), 2),
            "avg_client_payment_days": round(avg_client_days, 1),
            "avg_supplier_payment_days": 0,
            "monthly_burn_rate": round(monthly_burn, 2),
            "monthly_revenue_avg": round(monthly_revenue, 2),
            "revenue_recurring_pct": 0,
            "runway_months": round(min(runway, 99.0), 1),
            "top_expenses": [
                {"category": c, "total": round(a, 2), "monthly_avg": round(a / months, 2), "pct_of_total": round(a / total_expenses, 3) if total_expenses > 0 else 0}
                for c, a in top_exp
            ],
  }
