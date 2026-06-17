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
)

select
    timestamp_utc,
    timestamp_utc::date as date,
    extract(hour from timestamp_utc)::integer as hour,
    (extract(isodow from timestamp_utc)::integer - 1) as day_of_week,
    extract(month from timestamp_utc)::integer as month,
    extract(year from timestamp_utc)::integer as year,
    (extract(isodow from timestamp_utc)::integer in (6, 7)) as is_weekend
from calendar
