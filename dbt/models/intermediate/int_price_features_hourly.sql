select
    timestamp_utc,
    pool_price_cad_mwh,
    source,
    ingested_at,
    lag(pool_price_cad_mwh, 24) over (order by timestamp_utc) as price_lag_24h,
    avg(pool_price_cad_mwh) over (
        order by timestamp_utc
        rows between 23 preceding and current row
    ) as price_rolling_24h_avg
from {{ ref('stg_aeso_pool_price') }}
