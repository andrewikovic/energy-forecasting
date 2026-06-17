from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Holiday:
    name: str
    actual_date: date
    observed_date: date
    is_alberta_stat_holiday: bool
    is_canada_stat_holiday: bool


def _observed_fixed_holiday(actual_date: date) -> date:
    if actual_date.weekday() == 5:
        return actual_date + timedelta(days=2)
    if actual_date.weekday() == 6:
        return actual_date + timedelta(days=1)
    return actual_date


def _christmas_observed(year: int) -> date:
    christmas = date(year, 12, 25)
    if christmas.weekday() in {5, 6}:
        return date(year, 12, 27)
    return christmas


def _boxing_day_observed(year: int) -> date:
    boxing_day = date(year, 12, 26)
    if boxing_day.weekday() in {5, 6}:
        return date(year, 12, 28)
    return boxing_day


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first_day = date(year, month, 1)
    days_until_weekday = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_weekday + (n - 1) * 7)


def _monday_on_or_before(day: date) -> date:
    return day - timedelta(days=day.weekday())


def easter_date(year: int) -> date:
    """Return Gregorian Easter Sunday using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    leaping = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * leaping) // 451
    month = (h + leaping - 7 * m + 114) // 31
    day = ((h + leaping - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def holidays_for_year(year: int) -> tuple[Holiday, ...]:
    """Return Alberta and Canada statutory holiday dates for feature engineering.

    Observed dates are workday-style proxies. Collective agreements can differ,
    so source-specific overrides should be staged separately if needed.
    """
    easter = easter_date(year)
    fixed_canada_alberta = (
        ("New Year's Day", date(year, 1, 1), True, True),
        ("Canada Day", date(year, 7, 1), True, True),
        ("Remembrance Day", date(year, 11, 11), True, True),
    )
    rows = [
        Holiday(name, actual, _observed_fixed_holiday(actual), is_ab, is_ca)
        for name, actual, is_ab, is_ca in fixed_canada_alberta
    ]
    rows.extend(
        [
            Holiday(
                "Alberta Family Day",
                _nth_weekday_of_month(year, 2, weekday=0, n=3),
                _nth_weekday_of_month(year, 2, weekday=0, n=3),
                True,
                False,
            ),
            Holiday("Good Friday", easter - timedelta(days=2), easter - timedelta(days=2), True, True),
            Holiday(
                "Victoria Day",
                _monday_on_or_before(date(year, 5, 24)),
                _monday_on_or_before(date(year, 5, 24)),
                True,
                True,
            ),
            Holiday(
                "Labour Day",
                _nth_weekday_of_month(year, 9, weekday=0, n=1),
                _nth_weekday_of_month(year, 9, weekday=0, n=1),
                True,
                True,
            ),
            Holiday(
                "National Day for Truth and Reconciliation",
                date(year, 9, 30),
                _observed_fixed_holiday(date(year, 9, 30)),
                False,
                True,
            ),
            Holiday(
                "Thanksgiving Day",
                _nth_weekday_of_month(year, 10, weekday=0, n=2),
                _nth_weekday_of_month(year, 10, weekday=0, n=2),
                True,
                True,
            ),
            Holiday(
                "Christmas Day",
                date(year, 12, 25),
                _christmas_observed(year),
                True,
                True,
            ),
            Holiday(
                "Boxing Day",
                date(year, 12, 26),
                _boxing_day_observed(year),
                False,
                True,
            ),
        ]
    )
    return tuple(rows)


def holiday_flags(local_dates) -> dict[str, list[bool]]:
    years = {int(day.year) for day in local_dates}
    holiday_rows = [
        holiday
        for year in range(min(years) - 1, max(years) + 2)
        for holiday in holidays_for_year(year)
    ] if years else []

    alberta_dates = {
        holiday.observed_date for holiday in holiday_rows if holiday.is_alberta_stat_holiday
    }
    canada_dates = {
        holiday.observed_date for holiday in holiday_rows if holiday.is_canada_stat_holiday
    }
    long_weekend_dates: set[date] = set()
    for holiday in holiday_rows:
        observed_date = holiday.observed_date
        if observed_date.weekday() == 0:
            long_weekend_dates.update(
                observed_date - timedelta(days=offset) for offset in range(0, 4)
            )
        elif observed_date.weekday() == 4:
            long_weekend_dates.update(
                observed_date + timedelta(days=offset) for offset in range(0, 4)
            )

    is_alberta = [day in alberta_dates for day in local_dates]
    is_canada = [day in canada_dates for day in local_dates]
    is_stat = [ab or ca for ab, ca in zip(is_alberta, is_canada)]
    is_long_weekend = [day in long_weekend_dates for day in local_dates]
    is_local_weekend = [day.weekday() >= 5 for day in local_dates]
    is_non_workday = [
        weekend or holiday for weekend, holiday in zip(is_local_weekend, is_stat)
    ]

    return {
        "is_alberta_stat_holiday": is_alberta,
        "is_canada_stat_holiday": is_canada,
        "is_stat_holiday": is_stat,
        "is_long_weekend": is_long_weekend,
        "is_workday": [not value for value in is_non_workday],
        "is_non_workday": is_non_workday,
    }
