from enum import Enum


class Stance(str, Enum):
    bull = "bull"
    bear = "bear"
    neutral = "neutral"


class Horizon(str, Enum):
    intraday = "intraday"
    one_week = "1w"
    one_month = "1m"
    three_months = "3m"
    one_year = "1y"


class ExtractionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ReviewStatus(str, Enum):
    unreviewed = "unreviewed"
    approved = "approved"
    rejected = "rejected"


HORIZON_ORDER: list[str] = [
    Horizon.intraday.value,
    Horizon.one_week.value,
    Horizon.one_month.value,
    Horizon.three_months.value,
    Horizon.one_year.value,
]
