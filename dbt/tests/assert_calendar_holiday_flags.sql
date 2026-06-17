with expected as (
    select *
    from (values
        ('2024-01-01'::date, true, true, true, true, false),
        ('2024-01-02'::date, false, false, false, false, true),
        ('2024-02-19'::date, true, true, false, true, false),
        ('2024-03-29'::date, true, true, true, true, false),
        ('2024-07-01'::date, true, true, true, true, false)
    ) as rows(
        local_date,
        is_stat_holiday,
        is_alberta_stat_holiday,
        is_canada_stat_holiday,
        is_long_weekend,
        is_workday
    )
)

select
    calendar.local_date,
    calendar.is_stat_holiday,
    calendar.is_alberta_stat_holiday,
    calendar.is_canada_stat_holiday,
    calendar.is_long_weekend,
    calendar.is_workday
from {{ ref('stg_calendar') }} as calendar
inner join expected
    on calendar.local_date = expected.local_date
where calendar.is_stat_holiday is distinct from expected.is_stat_holiday
    or calendar.is_alberta_stat_holiday is distinct from expected.is_alberta_stat_holiday
    or calendar.is_canada_stat_holiday is distinct from expected.is_canada_stat_holiday
    or calendar.is_long_weekend is distinct from expected.is_long_weekend
    or calendar.is_workday is distinct from expected.is_workday
