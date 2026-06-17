with bounds as (
    select
        min(timestamp_utc) as min_timestamp_utc,
        max(timestamp_utc) as max_timestamp_utc
    from {{ ref('stg_aeso_load') }}
),

calendar as (
    select generate_series(
        min_timestamp_utc,
        max_timestamp_utc,
        interval '1 hour'
    ) as timestamp_utc
    from bounds
    where min_timestamp_utc is not null
),

calendar_local as (
    select
        timestamp_utc,
        timestamp_utc at time zone 'America/Edmonton' as timestamp_local
    from calendar
),

local_year_bounds as (
    select
        min(extract(year from timestamp_local)::integer) as min_year,
        max(extract(year from timestamp_local)::integer) as max_year
    from calendar_local
),

years as (
    select generate_series(min_year - 1, max_year + 1) as year
    from local_year_bounds
    where min_year is not null
),

easter_base as (
    select
        year,
        year % 19 as a,
        year / 100 as b,
        year % 100 as c
    from years
),

easter_parts as (
    select
        *,
        b / 4 as d,
        b % 4 as e,
        (b + 8) / 25 as f,
        c / 4 as i,
        c % 4 as k
    from easter_base
),

easter_calc as (
    select
        *,
        (b - f + 1) / 3 as g
    from easter_parts
),

easter_dates as (
    select
        year,
        make_date(
            year,
            ((h + l - 7 * m + 114) / 31)::integer,
            (((h + l - 7 * m + 114) % 31) + 1)::integer
        ) as easter_date
    from (
        select
            *,
            (a + 11 * h + 22 * l) / 451 as m
        from (
            select
                *,
                (32 + 2 * e + 2 * i - h - k) % 7 as l
            from (
                select
                    *,
                    (19 * a + b - d - g + 15) % 30 as h
                from easter_calc
            ) as h_calc
        ) as l_calc
    ) as m_calc
),

holiday_seed as (
    select
        year,
        'New Year''s Day' as holiday_name,
        make_date(year, 1, 1) as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Alberta Family Day' as holiday_name,
        (
            make_date(year, 2, 1)
            + (((8 - extract(isodow from make_date(year, 2, 1))::integer) % 7) * interval '1 day')
            + interval '14 days'
        )::date as actual_date,
        true as is_alberta_stat_holiday,
        false as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Good Friday' as holiday_name,
        (easter_date - interval '2 days')::date as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from easter_dates

    union all

    select
        year,
        'Victoria Day' as holiday_name,
        (
            make_date(year, 5, 24)
            - ((extract(isodow from make_date(year, 5, 24))::integer - 1) * interval '1 day')
        )::date as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Canada Day' as holiday_name,
        make_date(year, 7, 1) as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Labour Day' as holiday_name,
        (
            make_date(year, 9, 1)
            + (((8 - extract(isodow from make_date(year, 9, 1))::integer) % 7) * interval '1 day')
        )::date as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'National Day for Truth and Reconciliation' as holiday_name,
        make_date(year, 9, 30) as actual_date,
        false as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Thanksgiving Day' as holiday_name,
        (
            make_date(year, 10, 1)
            + (((8 - extract(isodow from make_date(year, 10, 1))::integer) % 7) * interval '1 day')
            + interval '7 days'
        )::date as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Remembrance Day' as holiday_name,
        make_date(year, 11, 11) as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Christmas Day' as holiday_name,
        make_date(year, 12, 25) as actual_date,
        true as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years

    union all

    select
        year,
        'Boxing Day' as holiday_name,
        make_date(year, 12, 26) as actual_date,
        false as is_alberta_stat_holiday,
        true as is_canada_stat_holiday
    from years
),

holiday_dates as (
    select
        holiday_name,
        actual_date,
        case
            when holiday_name = 'Christmas Day'
                and extract(isodow from actual_date)::integer in (6, 7)
                then make_date(year, 12, 27)
            when holiday_name = 'Boxing Day'
                and extract(isodow from actual_date)::integer in (6, 7)
                then make_date(year, 12, 28)
            when extract(isodow from actual_date)::integer = 6
                then (actual_date + interval '2 days')::date
            when extract(isodow from actual_date)::integer = 7
                then (actual_date + interval '1 day')::date
            else actual_date
        end as observed_date,
        is_alberta_stat_holiday,
        is_canada_stat_holiday
    from holiday_seed
),

holiday_by_date as (
    select
        observed_date as local_date,
        string_agg(holiday_name, ', ' order by holiday_name) as holiday_name,
        bool_or(is_alberta_stat_holiday) as is_alberta_stat_holiday,
        bool_or(is_canada_stat_holiday) as is_canada_stat_holiday
    from holiday_dates
    group by observed_date
),

long_weekend_dates as (
    select generate_series(
        observed_date - interval '3 days',
        observed_date,
        interval '1 day'
    )::date as local_date
    from holiday_dates
    where extract(isodow from observed_date)::integer = 1

    union

    select generate_series(
        observed_date,
        observed_date + interval '3 days',
        interval '1 day'
    )::date as local_date
    from holiday_dates
    where extract(isodow from observed_date)::integer = 5
)

select
    calendar_local.timestamp_utc,
    calendar_local.timestamp_utc::date as date,
    extract(hour from calendar_local.timestamp_utc)::integer as hour,
    (extract(isodow from calendar_local.timestamp_utc)::integer - 1) as day_of_week,
    extract(month from calendar_local.timestamp_utc)::integer as month,
    extract(year from calendar_local.timestamp_utc)::integer as year,
    (extract(isodow from calendar_local.timestamp_utc)::integer in (6, 7)) as is_weekend,
    calendar_local.timestamp_local::date as local_date,
    extract(hour from calendar_local.timestamp_local)::integer as local_hour,
    (extract(isodow from calendar_local.timestamp_local)::integer - 1) as local_day_of_week,
    (extract(isodow from calendar_local.timestamp_local)::integer in (6, 7)) as is_local_weekend,
    holidays.holiday_name,
    coalesce(holidays.is_alberta_stat_holiday, false) as is_alberta_stat_holiday,
    coalesce(holidays.is_canada_stat_holiday, false) as is_canada_stat_holiday,
    (
        coalesce(holidays.is_alberta_stat_holiday, false)
        or coalesce(holidays.is_canada_stat_holiday, false)
    ) as is_stat_holiday,
    long_weekends.local_date is not null as is_long_weekend,
    (
        extract(isodow from calendar_local.timestamp_local)::integer in (6, 7)
        or coalesce(holidays.is_alberta_stat_holiday, false)
        or coalesce(holidays.is_canada_stat_holiday, false)
    ) as is_non_workday,
    not (
        extract(isodow from calendar_local.timestamp_local)::integer in (6, 7)
        or coalesce(holidays.is_alberta_stat_holiday, false)
        or coalesce(holidays.is_canada_stat_holiday, false)
    ) as is_workday
from calendar_local
left join holiday_by_date as holidays
    on calendar_local.timestamp_local::date = holidays.local_date
left join long_weekend_dates as long_weekends
    on calendar_local.timestamp_local::date = long_weekends.local_date
