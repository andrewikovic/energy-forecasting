select
    timestamp_utc,
    load_mw,
    source,
    ingested_at,
    lag(load_mw, 1) over (order by timestamp_utc) as load_lag_1h,
    lag(load_mw, 24) over (order by timestamp_utc) as load_lag_24h,
    lag(load_mw, 168) over (order by timestamp_utc) as load_lag_168h,
    avg(load_mw) over (
        order by timestamp_utc
        rows between 23 preceding and current row
    ) as load_rolling_24h_avg,
    avg(load_mw) over (
        order by timestamp_utc
        rows between 167 preceding and current row
    ) as load_rolling_168h_avg
from {{ ref('stg_aeso_load') }}
