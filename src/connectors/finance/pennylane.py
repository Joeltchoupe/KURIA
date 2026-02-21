"""
Pennylane Connector — Comptabilité SaaS française, très populaire chez les PME FR.

API REST v1. Bearer token.
Rate limit : non documenté publiquement (safe : 60/min).

Pennylane structure :
- CustomerInvoices (factures clients)
- SupplierInvoices (factures fournisseurs)
- BankAccounts + BankTransactions
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
from connectors.utils import normalize_date, retry_with_backoff, safe_float


class PennylaneConnector(BaseConnector):

    CONNECTOR_NAME = "pennylane"
    CONNECTOR_CATEGORY = "finance"
    DATA_TYPES = ["accounts", "invoices", "bills", "summary"]

    BASE_URL = "https://app.pennylane.com/api/external/v1"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, api_token: str, **kwargs: Any) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/company")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid Pennylane token.")
            r.raise_for_status()
            self._authenticated = True
            self.logger.info("Pennylane authenticated")
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
            return (await self._client.get("/company")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            # Factures clients
            raw_invoices = await self._paginate("/customer_invoices", {"filter[date][gte]": since})
            invoices = [self._normalize_invoice(i) for i in raw_invoices]

            # Factures fournisseurs
            raw_bills = await self._paginate("/supplier_invoices", {"filter[date][gte]": since})
            bills = [self._normalize_bill(b) for b in raw_bills]

            # Comptes bancaires
            raw_accounts = await self._api_get("/bank_accounts")
            accounts = [
                {
                    "account_id": a.get("id", ""),
                    "name": a.get("name", ""),
                    "account_type": "BANK",
                    "balance": safe_float(a.get("balance"), 0),
                    "currency": a.get("currency", "EUR"),
                }
                for a in raw_accounts.get("bank_accounts", [])
            ]

            summary = self._calculate_summary(accounts, invoices, bills, days_back)
            metrics.complete(records=len(invoices) + len(bills) + len(accounts))

            return {
                "accounts": accounts,
                "invoices": invoices,
                "bills": bills,
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        if r.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, 60)
        if r.status_code == 401:
            raise AuthenticationError(self.CONNECTOR_NAME, "Token expired.")
        r.raise_for_status()
        return r.json()

    async def _paginate(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        all_items = []
        page = 1
        p = dict(params or {})

        while True:
            p["page"] = page
            p["per_page"] = 100
            data = await self._api_get(endpoint, p)

            # Pennylane wraps dans le nom de la ressource
            items = []
            for key in data:
                if isinstance(data[key], list):
                    items = data[key]
                    break

            if not items:
                break

            all_items.extend(items)
            # Check pagination
            pagination = data.get("pagination", {})
            if page >= pagination.get("total_pages", 1):
                break
            page += 1

        return all_items

    def _normalize_invoice(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        amount = safe_float(raw.get("amount"), 0)
        amount_due = safe_float(raw.get("remaining_amount", raw.get("amount")), 0)
        due_date = normalize_date(raw.get("deadline"))
        issue_date = normalize_date(raw.get("date"))

        is_overdue = False
        days_overdue = 0
        if due_date and amount_due > 0:
            is_overdue = now > due_date
            if is_overdue:
                days_overdue = (now - due_date).days

        paid = raw.get("status", "").lower() in ("paid", "payée")
        status = "paid" if paid else ("overdue" if is_overdue else "pending")

        days_to_payment = None
        if paid and issue_date:
            paid_date = normalize_date(raw.get("paid_at"))
            if paid_date:
                days_to_payment = max(0, (paid_date - issue_date).days)

        return {
            "invoice_id": str(raw.get("id", "")),
            "customer_name": raw.get("customer", {}).get("name", ""),
            "amount": amount,
            "amount_due": 0 if paid else amount_due,
            "issue_date": issue_date,
            "due_date": due_date,
            "status": status,
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "days_to_payment": days_to_payment,
            "currency": raw.get("currency", "EUR"),
        }

    def _normalize_bill(self, raw: dict) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        amount = safe_float(raw.get("amount"), 0)
        amount_due = safe_float(raw.get("remaining_amount", raw.get("amount")), 0)
        due_date = normalize_date(raw.get("deadline"))

        paid = raw.get("status", "").lower() in ("paid", "payée")
        is_overdue = False
        if due_date and not paid and amount_due > 0:
            is_overdue = now > due_date

        status = "paid" if paid else ("overdue" if is_overdue else "pending")

        category = "uncategorized"
        lines = raw.get("line_items", [])
        if lines:
            category = lines[0].get("account", {}).get("label", "uncategorized")

        return {
            "bill_id": str(raw.get("id", "")),
            "vendor_name": raw.get("supplier", {}).get("name", ""),
            "amount": amount,
            "amount_due": 0 if paid else amount_due,
            "issue_date": normalize_date(raw.get("date")),
            "due_date": due_date,
            "status": status,
            "is_overdue": is_overdue,
            "category": category,
            "currency": raw.get("currency", "EUR"),
        }

    def _calculate_summary(self, accounts, invoices, bills, days_back) -> dict[str, Any]:
        """Même logique que QuickBooks/Xero — format de sortie identique."""
        cash_position = sum(a.get("balance", 0) for a in accounts)
        months = max(1, days_back / 30)

        total_receivable = sum(i.get("amount_due", 0) for i in invoices if i["status"] != "paid")
        total_payable = sum(b.get("amount_due", 0) for b in bills if b["status"] != "paid")

        overdue = [i for i in invoices if i.get("is_overdue")]
        overdue_count = len(overdue)
        overdue_value = sum(i.get("amount_due", 0) for i in overdue)

        payment_days = [i["days_to_payment"] for i in invoices if i.get("days_to_payment") and i["days_to_payment"] > 0]
        avg_client_days = sum(payment_days) / len(payment_days) if payment_days else 0

        total_expenses = sum(b.get("amount", 0) for b in bills)
        total_revenue = sum(i.get("amount", 0) for i in invoices)
        monthly_burn = total_expenses / months
        monthly_revenue = total_revenue / months

        net = monthly_revenue - monthly_burn
        runway = 99.0
        if net < 0:
            runway = cash_position / abs(net) if net != 0 else 0

        return {
            "cash_position": round(cash_position, 2),
            "total_receivable": round(total_receivable, 2),
            "total_payable": round(total_payable, 2),
            "invoices_overdue_count": overdue_count,
            "invoices_overdue_value": round(overdue_value, 2),
            "avg_client_payment_days": round(avg_client_days, 1),
            "avg_supplier_payment_days": 0,
            "monthly_burn_rate": round(monthly_burn, 2),
            "monthly_revenue_avg": round(monthly_revenue, 2),
            "revenue_recurring_pct": 0,
            "runway_months": round(min(runway, 99.0), 1),
            "top_expenses": [],
        }
