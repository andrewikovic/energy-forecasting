from datetime import date

from aeso_analytics.holidays import easter_date, holiday_flags, holidays_for_year


def test_easter_date_is_deterministic_for_known_years():
    assert easter_date(2024) == date(2024, 3, 31)
    assert easter_date(2025) == date(2025, 4, 20)


def test_holidays_for_year_includes_alberta_and_canada_observed_dates():
    holidays = holidays_for_year(2024)
    by_name = {holiday.name: holiday for holiday in holidays}

    assert by_name["Alberta Family Day"].observed_date == date(2024, 2, 19)
    assert by_name["Good Friday"].observed_date == date(2024, 3, 29)
    assert by_name["Canada Day"].observed_date == date(2024, 7, 1)
    assert by_name["National Day for Truth and Reconciliation"].is_canada_stat_holiday
    assert not by_name["National Day for Truth and Reconciliation"].is_alberta_stat_holiday


def test_holiday_flags_mark_long_weekends_and_workdays():
    flags = holiday_flags([date(2024, 1, 1), date(2024, 1, 2), date(2024, 3, 29)])

    assert flags["is_stat_holiday"] == [True, False, True]
    assert flags["is_long_weekend"] == [True, False, True]
    assert flags["is_workday"] == [False, True, False]
    assert flags["is_non_workday"] == [True, False, True]
