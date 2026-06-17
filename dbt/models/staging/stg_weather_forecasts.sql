{% set raw_weather_relation = adapter.get_relation(
    database=target.database,
    schema='raw',
    identifier='raw_weather_forecast_hourly'
) %}

{% if not var('enable_weather_features', true) or raw_weather_relation is none %}

select
    cast(null as timestamptz) as forecast_issue_utc,
    cast(null as timestamptz) as forecast_target_utc,
    cast(null as integer) as forecast_horizon_hours,
    cast(null as text) as region,
    cast(null as numeric(8, 2)) as temperature_c,
    cast(null as numeric(8, 2)) as wind_speed_mps,
    cast(null as numeric(8, 2)) as relative_humidity_pct,
    cast(null as numeric(10, 3)) as precipitation_mm,
    cast(null as numeric(8, 2)) as heating_degree_c,
    cast(null as numeric(8, 2)) as cooling_degree_c,
    cast(null as text) as source,
    cast(null as timestamptz) as ingested_at
where false

{% else %}

with source_rows as (
    select
        forecast_issue_utc::timestamptz as forecast_issue_utc,
        forecast_target_utc::timestamptz as forecast_target_utc,
        lower(coalesce(nullif(trim(region::text), ''), 'alberta')) as region,
        temperature_c::numeric(8, 2) as temperature_c,
        wind_speed_mps::numeric(8, 2) as wind_speed_mps,
        relative_humidity_pct::numeric(8, 2) as relative_humidity_pct,
        precipitation_mm::numeric(10, 3) as precipitation_mm,
        source::text as source,
        ingested_at::timestamptz as ingested_at
    from {{ raw_weather_relation }}
    where forecast_issue_utc is not null
        and forecast_target_utc is not null
        and forecast_target_utc >= forecast_issue_utc
),

deduped as (
    select
        *,
        row_number() over (
            partition by forecast_issue_utc, forecast_target_utc, region
            order by ingested_at desc nulls last
        ) as row_number
    from source_rows
)

select
    forecast_issue_utc,
    forecast_target_utc,
    (extract(epoch from (forecast_target_utc - forecast_issue_utc)) / 3600)::integer
        as forecast_horizon_hours,
    region,
    temperature_c,
    wind_speed_mps,
    relative_humidity_pct,
    precipitation_mm,
    case
        when temperature_c is null then null
        else greatest(18.0 - temperature_c, 0)
    end::numeric(8, 2) as heating_degree_c,
    case
        when temperature_c is null then null
        else greatest(temperature_c - 18.0, 0)
    end::numeric(8, 2) as cooling_degree_c,
    source,
    ingested_at
from deduped
where row_number = 1

{% endif %}
