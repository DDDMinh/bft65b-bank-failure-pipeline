"""Small date helpers used by configuration and later label/split modules."""

from __future__ import annotations

from datetime import date, datetime


def parse_date(value: str | date | datetime) -> date:
    """Parse an ISO date or accept an existing date/datetime."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def quarter_end(value: str | date | datetime) -> date:
    """Return the quarter-end date containing ``value``."""
    parsed = parse_date(value)
    quarter_month = ((parsed.month - 1) // 3 + 1) * 3
    if quarter_month == 3:
        return date(parsed.year, 3, 31)
    if quarter_month == 6:
        return date(parsed.year, 6, 30)
    if quarter_month == 9:
        return date(parsed.year, 9, 30)
    return date(parsed.year, 12, 31)


def quarter_index(value: str | date | datetime) -> int:
    """Map a date to a monotonically increasing quarter index."""
    parsed = quarter_end(value)
    return parsed.year * 4 + (parsed.month - 1) // 3


def add_quarters(value: str | date | datetime, quarters: int) -> date:
    """Add a whole number of quarters and return the resulting quarter end."""
    parsed = quarter_end(value)
    index = quarter_index(parsed) + quarters
    year, zero_based_quarter = divmod(index, 4)
    month = (zero_based_quarter + 1) * 3
    return quarter_end(date(year, month, 1))


def quarters_between(start: str | date | datetime, end: str | date | datetime) -> int:
    """Number of whole quarter steps from ``start`` to ``end``."""
    return quarter_index(end) - quarter_index(start)


def quarter_ends_between(
    start: str | date | datetime, end: str | date | datetime
) -> list[date]:
    """Return every quarter-end date in ``[start, end]`` inclusive."""
    first = quarter_end(start)
    last = quarter_end(end)
    if first > last:
        raise ValueError(f"Start quarter {first} is after end quarter {last}")
    quarters: list[date] = []
    current = first
    while current <= last:
        quarters.append(current)
        current = add_quarters(current, 1)
    return quarters

