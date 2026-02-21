"""
CSV Fallback Connector — Pour les clients sans API.

40-50% des PME n'ont pas HubSpot.
Certaines utilisent Pipedrive, Zoho, un Excel, ou rien.
Ce connecteur accepte des fichiers CSV et les normalise
vers les mêmes structures que les connecteurs API.

Design decisions :
- Pandas pour le parsing (robuste, gère les encodages et séparateurs)
- Détection automatique des colonnes (mapping intelligent)
- Validation stricte : si les colonnes critiques manquent, on refuse
- Même format de sortie que HubSpot/Gmail/QuickBooks
  → Les agents ne savent pas d'où viennent les données
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from connectors.base import BaseConnector, ConnectorError
from connectors.utils import normalize_date, safe_float, safe_int, truncate

logger = logging.getLogger("kuria.connectors.csv_fallback")


# ──────────────────────────────────────────────
# COLUMN DETECTION
# ──────────────────────────────────────────────

# Mapping de noms de colonnes courants vers nos noms standards.
# Chaque clé standard a une liste de noms possibles (lowercase, stripped).
# On matche le premier trouvé.

DEAL_COLUMN_MAPPINGS: dict[str, list[str]] = {
    "deal_id": ["id", "deal_id", "dealid", "deal id", "opportunity_id", "opp_id"],
    "name": ["name", "deal_name", "dealname", "deal name", "opportunity", "titre", "title"],
    "stage": ["stage", "deal_stage", "dealstage", "deal stage", "étape", "etape", "status"],
    "amount": ["amount", "montant", "value", "valeur", "deal_amount", "revenue", "ca"],
    "close_date": [
        "close_date", "closedate", "close date", "date_close",
        "expected_close", "date fermeture", "date de clôture",
    ],
    "create_date": [
        "create_date", "createdate", "created", "created_at",
        "date_created", "creation_date", "date création", "date de création",
    ],
    "owner": ["owner", "owner_name", "commercial", "sales_rep", "rep", "vendeur", "assigné"],
    "source": ["source", "lead_source", "origine", "canal", "channel"],
    "is_won": ["is_won", "won", "gagné", "gagne", "closed_won", "status_won"],
    "is_closed": ["is_closed", "closed", "fermé", "ferme", "clos"],
}

CONTACT_COLUMN_MAPPINGS: dict[str, list[str]] = {
    "contact_id": ["id", "contact_id", "contactid"],
    "email": ["email", "e-mail", "mail", "adresse_email", "email_address"],
    "first_name": ["first_name", "firstname", "prénom", "prenom", "first"],
    "last_name": ["last_name", "lastname", "nom", "last", "family_name"],
    "company": ["company", "entreprise", "société", "societe", "organization"],
    "source": ["source", "lead_source", "origine", "canal"],
    "create_date": ["create_date", "created", "created_at", "date_created"],
}

TRANSACTION_COLUMN_MAPPINGS: dict[str, list[str]] = {
    "transaction_id": ["id", "transaction_id", "txn_id", "ref", "reference"],
    "date": ["date", "txn_date", "transaction_date", "date_facture"],
    "amount": ["amount", "montant", "total", "total_amount", "total_ttc"],
    "type": ["type", "transaction_type", "txn_type"],
    "category": ["category", "catégorie", "categorie", "compte", "account"],
    "description": ["description", "libellé", "libelle", "memo", "note"],
    "status": ["status", "statut", "état", "etat"],
    "counterpart": [
        "customer", "client", "vendor", "fournisseur",
        "counterpart", "tiers", "nom",
    ],
    "due_date": ["due_date", "echeance", "échéance", "date_echeance"],
}


# ──────────────────────────────────────────────
# CSV CONNECTOR
# ──────────────────────────────────────────────


class CSVConnector(BaseConnector):
    """Connecteur CSV — fallback pour les outils sans API.

    Usage :
        connector = CSVConnector(company_id="acme")
        await connector.authenticate()
        data = await connector.load_deals_csv("path/to/deals.csv")
    """

    CONNECTOR_NAME = "csv_fallback"

    # Encodages à essayer dans l'ordre
    ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]

    # Séparateurs à essayer
    SEPARATORS = [",", ";", "\t", "|"]

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._loaded_data: dict[str, Any] = {}

    async def authenticate(self, **kwargs: Any) -> None:
        """Le CSV n'a pas besoin d'auth. On marque juste comme prêt."""
        self._authenticated = True
        self.logger.info("CSV connector ready")

    async def health_check(self) -> bool:
        return self._authenticated

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        """Retourne les données déjà chargées via load_*_csv()."""
        return self._loaded_data

    # ── Loaders ──

    async def load_deals_csv(
        self, filepath: Union[str, Path]
    ) -> list[dict[str, Any]]:
        """Charge et normalise un CSV de deals/opportunités.

        Returns:
            Liste de dicts au même format que HubSpotConnector.extract()["deals"]
        """
        df = self._read_csv(filepath)
        mapping = self._detect_columns(df, DEAL_COLUMN_MAPPINGS)

        self.logger.info(
            f"Detected columns: {mapping} from {list(df.columns)}"
        )

        # Vérifier les colonnes critiques
        if "name" not in mapping and "deal_id" not in mapping:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=(
                    "Cannot identify deal name or ID column. "
                    f"Available columns: {list(df.columns)}"
                ),
                recoverable=False,
            )

        deals = []
        now = datetime.now(tz=timezone.utc)

        for idx, row in df.iterrows():
            create_date = normalize_date(
                row.get(mapping.get("create_date", ""), None)
            )
            close_date = normalize_date(
                row.get(mapping.get("close_date", ""), None)
            )

            days_since_activity = 0
            if create_date:
                days_since_activity = (now - create_date).days

            # Détecter is_won / is_closed
            is_won = False
            is_closed = False
            if "is_won" in mapping:
                won_val = str(row.get(mapping["is_won"], "")).lower().strip()
                is_won = won_val in ("true", "1", "yes", "oui", "gagné", "gagne", "won")
            if "is_closed" in mapping:
                closed_val = str(row.get(mapping["is_closed"], "")).lower().strip()
                is_closed = closed_val in (
                    "true", "1", "yes", "oui", "fermé", "ferme", "closed",
                )
            if is_won:
                is_closed = True

            deal = {
                "deal_id": str(
                    row.get(mapping.get("deal_id", ""), idx)
                ),
                "name": truncate(
                    str(row.get(mapping.get("name", ""), f"Deal {idx}")),
                    max_length=256,
                ),
                "stage": str(row.get(mapping.get("stage", ""), "unknown")),
                "pipeline": "default",
                "amount": safe_float(
                    row.get(mapping.get("amount", ""), 0)
                ),
                "close_date": close_date,
                "create_date": create_date,
                "owner_id": None,
                "owner_name": str(
                    row.get(mapping.get("owner", ""), "Unknown")
                ),
                "is_closed": is_closed,
                "is_won": is_won,
                "days_in_current_stage": days_since_activity,
                "days_since_last_activity": days_since_activity,
                "last_activity_date": create_date,
                "last_activity_type": None,
                "activity_count_30d": 0,
                "has_next_step": False,
                "contact_ids": [],
                "source": str(
                    row.get(mapping.get("source", ""), "")
                ) or None,
                "raw_properties": row.to_dict(),
            }
            deals.append(deal)

        self._loaded_data["deals"] = deals
        self.logger.info(f"Loaded {len(deals)} deals from CSV")
        return deals

    async def load_contacts_csv(
        self, filepath: Union[str, Path]
    ) -> list[dict[str, Any]]:
        """Charge et normalise un CSV de contacts."""
        df = self._read_csv(filepath)
        mapping = self._detect_columns(df, CONTACT_COLUMN_MAPPINGS)

        if "email" not in mapping and "last_name" not in mapping:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=(
                    "Cannot identify email or name column. "
                    f"Available columns: {list(df.columns)}"
                ),
                recoverable=False,
            )

        contacts = []
        for idx, row in df.iterrows():
            contact = {
                "contact_id": str(
                    row.get(mapping.get("contact_id", ""), idx)
                ),
                "email": str(
                    row.get(mapping.get("email", ""), "")
                ) or None,
                "first_name": str(
                    row.get(mapping.get("first_name", ""), "")
                ) or None,
                "last_name": str(
                    row.get(mapping.get("last_name", ""), "")
                ) or None,
                "company": str(
                    row.get(mapping.get("company", ""), "")
                ) or None,
                "job_title": None,
                "lifecycle_stage": None,
                "lead_status": None,
                "source": str(
                    row.get(mapping.get("source", ""), "")
                ) or None,
                "source_detail": None,
                "create_date": normalize_date(
                    row.get(mapping.get("create_date", ""), None)
                ),
                "last_activity_date": None,
                "owner_id": None,
                "associated_deal_ids": [],
            }
            contacts.append(contact)

        self._loaded_data["contacts"] = contacts
        self.logger.info(f"Loaded {len(contacts)} contacts from CSV")
        return contacts

    async def load_transactions_csv(
        self, filepath: Union[str, Path]
    ) -> dict[str, Any]:
        """Charge et normalise un CSV de transactions financières.

        Détecte automatiquement les factures (invoices) et les dépenses (bills)
        basé sur le signe du montant ou la colonne type.
        """
        df = self._read_csv(filepath)
        mapping = self._detect_columns(df, TRANSACTION_COLUMN_MAPPINGS)

        if "amount" not in mapping:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=(
                    "Cannot identify amount column. "
                    f"Available columns: {list(df.columns)}"
                ),
                recoverable=False,
            )

        invoices = []
        bills = []
        now = datetime.now(tz=timezone.utc)

        for idx, row in df.iterrows():
            amount = safe_float(row.get(mapping.get("amount", ""), 0))
            txn_date = normalize_date(
                row.get(mapping.get("date", ""), None)
            )
            due_date = normalize_date(
                row.get(mapping.get("due_date", ""), None)
            )

            # Détecter le type
            txn_type = str(
                row.get(mapping.get("type", ""), "")
            ).lower().strip()

            is_revenue = (
                amount > 0
                or txn_type in ("invoice", "facture", "vente", "revenu", "revenue", "income")
            )

            counterpart = str(
                row.get(mapping.get("counterpart", ""), "Unknown")
            )
            category = str(
                row.get(mapping.get("category", ""), "uncategorized")
            )
            status_raw = str(
                row.get(mapping.get("status", ""), "")
            ).lower().strip()

            abs_amount = abs(amount)

            if is_revenue:
                is_overdue = False
                days_overdue = 0
                if due_date:
                    is_overdue = now > due_date and status_raw not in ("paid", "payé", "payée")
                    if is_overdue:
                        days_overdue = (now - due_date).days

                status = "paid"
                if status_raw in ("pending", "en attente", "envoyé", "envoyée"):
                    status = "pending"
                elif is_overdue:
                    status = "overdue"

                invoices.append({
                    "invoice_id": str(
                        row.get(mapping.get("transaction_id", ""), f"INV-{idx}")
                    ),
                    "customer_name": counterpart,
                    "amount": abs_amount,
                    "amount_due": abs_amount if status != "paid" else 0,
                    "issue_date": txn_date,
                    "due_date": due_date,
                    "status": status,
                    "is_overdue": is_overdue,
                    "days_overdue": days_overdue,
                    "days_to_payment": None,
                    "currency": "EUR",
                })
            else:
                bills.append({
                    "bill_id": str(
                        row.get(mapping.get("transaction_id", ""), f"BILL-{idx}")
                    ),
                    "vendor_name": counterpart,
                    "amount": abs_amount,
                    "amount_due": abs_amount if status_raw not in ("paid", "payé") else 0,
                    "issue_date": txn_date,
                    "due_date": due_date,
                    "status": "paid" if status_raw in ("paid", "payé") else "pending",
                    "is_overdue": False,
                    "category": category,
                    "currency": "EUR",
                })

        self._loaded_data["invoices"] = invoices
        self._loaded_data["bills"] = bills
        self.logger.info(
            f"Loaded {len(invoices)} invoices and {len(bills)} bills from CSV"
        )
        return {"invoices": invoices, "bills": bills}

    # ── Internal helpers ──

    def _read_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Lit un CSV en essayant plusieurs encodages et séparateurs.

        Les PME françaises produisent des CSV dans tous les formats possibles.
        Cette méthode essaie tout jusqu'à ce que ça marche.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"File not found: {filepath}",
                recoverable=False,
            )

        if filepath.stat().st_size == 0:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"File is empty: {filepath}",
                recoverable=False,
            )

        last_error: Optional[Exception] = None

        for encoding in self.ENCODINGS:
            for sep in self.SEPARATORS:
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding=encoding,
                        sep=sep,
                        dtype=str,  # Tout en string, on convertit après
                        na_values=["", "N/A", "n/a", "NA", "null", "NULL", "-"],
                        keep_default_na=True,
                    )

                    # Vérifier que le parsing a produit quelque chose de sensé
                    if len(df.columns) < 2:
                        continue  # Probablement le mauvais séparateur
                    if len(df) == 0:
                        continue

                    # Nettoyer les noms de colonnes
                    df.columns = [
                        col.strip().lower().replace(" ", "_")
                        for col in df.columns
                    ]

                    self.logger.info(
                        f"Read CSV: {len(df)} rows, {len(df.columns)} cols, "
                        f"encoding={encoding}, sep='{sep}'"
                    )
                    return df

                except Exception as e:
                    last_error = e
                    continue

        raise ConnectorError(
            connector_name=self.CONNECTOR_NAME,
            message=(
                f"Failed to parse CSV with any encoding/separator combination. "
                f"Last error: {last_error}"
            ),
            recoverable=False,
            raw_error=last_error,
        )

    @staticmethod
    def _detect_columns(
        df: pd.DataFrame,
        mappings: dict[str, list[str]],
    ) -> dict[str, str]:
        """Détecte automatiquement les colonnes du CSV.

        Pour chaque champ standard, cherche dans la liste des noms possibles.
        Retourne un mapping {standard_name: actual_column_name}.
        """
        detected: dict[str, str] = {}
        df_columns_lower = {col.lower().strip(): col for col in df.columns}

        for standard_name, possible_names in mappings.items():
            for possible in possible_names:
                possible_clean = possible.lower().strip()
                if possible_clean in df_columns_lower:
                    detected[standard_name] = df_columns_lower[possible_clean]
                    break

        return detected
