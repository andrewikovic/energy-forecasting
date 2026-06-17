select
    timestamp_utc,
    date,
    hour,
    day_of_week,
    month,
    year,
    is_weekend,
    load_mw,
    pool_price_cad_mwh,
    avg(load_mw) over (partition by date) as daily_average_load_mw,
    max(load_mw) over (partition by date) as daily_peak_load_mw,
    avg(pool_price_cad_mwh) over (partition by date) as daily_average_pool_price_cad_mwh,
    load_rolling_24h_avg,
    load_rolling_168h_avg,
    price_rolling_24h_avg,
    latest_ingestion_time
from {{ ref('int_energy_features_hourly') }}
where load_mw is not null
