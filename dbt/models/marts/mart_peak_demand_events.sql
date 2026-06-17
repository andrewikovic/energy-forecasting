with scored as (
    select
        timestamp_utc,
        date,
        hour,
        day_of_week,
        month,
        load_mw,
        pool_price_cad_mwh,
        cume_dist() over (order by load_mw) as load_cume_dist,
        row_number() over (order by load_mw desc, timestamp_utc desc) as peak_rank
    from {{ ref('int_energy_features_hourly') }}
    where load_mw is not null
)

select
    timestamp_utc,
    date,
    hour,
    day_of_week,
    month,
    load_mw,
    pool_price_cad_mwh,
    load_cume_dist,
    peak_rank,
    true as is_top_5pct_load
from scored
where load_cume_dist >= 0.95
