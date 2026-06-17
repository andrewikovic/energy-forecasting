select
    calendar.timestamp_utc,
    calendar.date,
    calendar.hour,
    calendar.day_of_week,
    calendar.month,
    calendar.year,
    calendar.is_weekend,
    load_features.load_mw,
    load_features.load_lag_1h,
    load_features.load_lag_24h,
    load_features.load_lag_168h,
    load_features.load_rolling_24h_avg,
    load_features.load_rolling_168h_avg,
    price_features.pool_price_cad_mwh,
    price_features.price_lag_24h,
    price_features.price_rolling_24h_avg,
    greatest(load_features.ingested_at, price_features.ingested_at) as latest_ingestion_time
from {{ ref('stg_calendar') }} as calendar
left join {{ ref('int_load_features_hourly') }} as load_features
    on calendar.timestamp_utc = load_features.timestamp_utc
left join {{ ref('int_price_features_hourly') }} as price_features
    on calendar.timestamp_utc = price_features.timestamp_utc
