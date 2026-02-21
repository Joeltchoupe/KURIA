"""
Utilitaires partagés par tous les connecteurs.
V2 : enrichi avec des helpers pour calendrier, chat, marketing.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence, TypeVar

from dateutil import parser as dateutil_parser

logger = logging.getLogger("kuria.connectors.utils")
T = TypeVar("T")


# ──────────────────────────────────────────────
# RETRY (inchangé)
# ──────────────────────────────────────────────


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(
                            f"[retry] {func.__name__} failed after "
                            f"{max_retries} attempts: {e}"
                        )
                        raise
                    backoff = min(
                        max_delay,
                        base_delay * (exponential_base ** attempt),
                    )
                    jittered = random.uniform(0, backoff)
                    logger.warning(
                        f"[retry] {func.__name__} attempt {attempt + 1}/{max_retries} "
                        f"failed: {e}. Retrying in {jittered:.1f}s..."
                    )
                    await asyncio.sleep(jittered)
            raise last_exception
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# DATE NORMALIZATION (inchangé)
# ──────────────────────────────────────────────


def normalize_date(
    value: Any,
    default: Optional[datetime] = None,
) -> Optional[datetime]:
    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    if isinstance(value, str) and not value.strip():
        return default
    if isinstance(value, (int, float)):
        try:
            if value > 1e12:
                value = value / 1000
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return default
    if isinstance(value, str):
        try:
            parsed = dateutil_parser.parse(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            return default
    return default


# ──────────────────────────────────────────────
# SAFE TYPE CONVERSIONS (inchangé)
# ──────────────────────────────────────────────


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if not value.strip():
            return default
        cleaned = value.strip()
        for char in ("€", "$", "£", "\u00a0", " ", "%"):
            cleaned = cleaned.replace(char, "")
        has_comma = "," in cleaned
        has_dot = "." in cleaned
        if has_comma and has_dot:
            last_comma = cleaned.rfind(",")
            last_dot = cleaned.rfind(".")
            if last_comma > last_dot:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif has_comma and not has_dot:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


# ──────────────────────────────────────────────
# COLLECTIONS (inchangé)
# ──────────────────────────────────────────────


def chunk_list(lst: Sequence[T], size: int) -> list[list[T]]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [list(lst[i: i + size]) for i in range(0, len(lst), size)]


# ──────────────────────────────────────────────
# STRING / EMAIL (inchangé)
# ──────────────────────────────────────────────


def clean_email(email: Optional[str]) -> Optional[str]:
    if not email or not isinstance(email, str):
        return None
    cleaned = email.strip().lower()
    if "@" not in cleaned or "." not in cleaned:
        return None
    return cleaned


def extract_domain(email: Optional[str]) -> Optional[str]:
    cleaned = clean_email(email)
    if not cleaned:
        return None
    try:
        return cleaned.split("@")[1]
    except IndexError:
        return None


def truncate(text: Optional[str], max_length: int = 500) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


# ──────────────────────────────────────────────
# TIME / CALENDAR (enrichi V2)
# ──────────────────────────────────────────────


def is_business_hours(
    dt: datetime,
    start_hour: int = 8,
    end_hour: int = 20,
) -> bool:
    return start_hour <= dt.hour < end_hour


def is_business_day(dt: datetime) -> bool:
    return dt.weekday() < 5


def calculate_business_days(start: datetime, end: datetime) -> int:
    if end < start:
        return 0
    business_days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            business_days += 1
        current += timedelta(days=1)
    return business_days


def meeting_duration_minutes(start: datetime, end: datetime) -> float:
    """Calcule la durée d'un meeting en minutes."""
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 60


def hours_in_meetings_per_week(
    meetings: list[dict[str, Any]],
    weeks: int = 1,
) -> float:
    """Calcule le nombre d'heures en réunion par semaine."""
    total_minutes = 0.0
    for m in meetings:
        start = normalize_date(m.get("start"))
        end = normalize_date(m.get("end"))
        if start and end:
            total_minutes += meeting_duration_minutes(start, end)
    weeks = max(1, weeks)
    return round(total_minutes / 60 / weeks, 1)


# ──────────────────────────────────────────────
# MARKETING METRICS (nouveau V2)
# ──────────────────────────────────────────────


def calculate_ctr(clicks: int, impressions: int) -> float:
    """Click-through rate."""
    if impressions == 0:
        return 0.0
    return round(clicks / impressions, 4)


def calculate_cpc(spend: float, clicks: int) -> float:
    """Cost per click."""
    if clicks == 0:
        return 0.0
    return round(spend / clicks, 2)


def calculate_cpm(spend: float, impressions: int) -> float:
    """Cost per mille (1000 impressions)."""
    if impressions == 0:
        return 0.0
    return round(spend / impressions * 1000, 2)


def calculate_roas(revenue: float, spend: float) -> float:
    """Return on ad spend."""
    if spend == 0:
        return 0.0
    return round(revenue / spend, 2)


# ──────────────────────────────────────────────
# CHAT / SUPPORT METRICS (nouveau V2)
# ──────────────────────────────────────────────


def calculate_response_time_hours(
    messages: list[dict[str, Any]],
    time_field: str = "timestamp",
) -> float:
    """Calcule le temps de réponse moyen entre messages consécutifs."""
    if len(messages) < 2:
        return 0.0

    response_times = []
    sorted_msgs = sorted(
        messages,
        key=lambda m: normalize_date(m.get(time_field)) or datetime.min.replace(tzinfo=timezone.utc),
    )

    for i in range(1, len(sorted_msgs)):
        prev_time = normalize_date(sorted_msgs[i - 1].get(time_field))
        curr_time = normalize_date(sorted_msgs[i].get(time_field))
        if prev_time and curr_time:
            delta_hours = (curr_time - prev_time).total_seconds() / 3600
            if 0 < delta_hours < 720:
                response_times.append(delta_hours)

    if not response_times:
        return 0.0
    return round(sum(response_times) / len(response_times), 2)


def classify_ticket_priority(
    subject: str,
    body: str = "",
) -> str:
    """Heuristique simple pour classifier la priorité d'un ticket support."""
    text = f"{subject} {body}".lower()
    urgent_keywords = [
        "urgent", "critical", "down", "broken", "crash",
        "panne", "bloqué", "bloquant", "grave", "p0", "p1",
    ]
    high_keywords = [
        "bug", "error", "erreur", "problème", "problem",
        "issue", "cannot", "impossible", "échec",
    ]
    if any(kw in text for kw in urgent_keywords):
        return "urgent"
    if any(kw in text for kw in high_keywords):
        return "high"
    return "normal"
