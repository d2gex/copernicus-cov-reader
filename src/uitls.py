from datetime import date, timedelta
from typing import List, Tuple


def month_periods(year: int) -> List[Tuple[str, str]]:
    """Return (start, end) date strings for each month in the given year."""
    periods = []
    for month in range(1, 13):
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        periods.append((start.isoformat(), end.isoformat()))
    return periods


def year_to_month_sequence(years: List[int]) -> List[int]:
    """Return (start, end) date strings for each month in each given year in the given list."""
    periods = []
    for year in sorted(years):
        periods.extend(month_periods(year))
    return periods
