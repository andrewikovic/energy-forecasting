select
    forecast_issue_utc,
    forecast_target_utc,
    forecast_horizon_hours,
    count(*) as weather_forecast_region_count,
    avg(temperature_c)::numeric(8, 2) as forecast_temperature_c,
    avg(wind_speed_mps)::numeric(8, 2) as forecast_wind_speed_mps,
    avg(relative_humidity_pct)::numeric(8, 2) as forecast_relative_humidity_pct,
    avg(precipitation_mm)::numeric(10, 3) as forecast_precipitation_mm,
    avg(heating_degree_c)::numeric(8, 2) as forecast_heating_degree_c,
    avg(cooling_degree_c)::numeric(8, 2) as forecast_cooling_degree_c,
    max(ingested_at) as latest_weather_ingested_at
from {{ ref('stg_weather_forecasts') }}
group by
    forecast_issue_utc,
    forecast_target_utc,
    forecast_horizon_hours
