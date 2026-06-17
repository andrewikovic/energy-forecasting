with horizons as (
    select 1 as horizon_hours
    union all
    select 24 as horizon_hours
),

target_hours as (
    select * from {{ ref('int_energy_features_hourly') }}
    where load_mw is not null
),

training_rows as (
    select
        md5(concat(target.timestamp_utc::text, ':', horizons.horizon_hours::text)) as training_row_id,
        horizons.horizon_hours,
        issue.timestamp_utc as feature_as_of_timestamp_utc,
        greatest(
            issue.timestamp_utc,
            coalesce(weather.forecast_issue_utc, issue.timestamp_utc),
            coalesce(generation.availability_issue_utc, issue.timestamp_utc),
            coalesce(intertie.schedule_issue_utc, issue.timestamp_utc)
        ) as feature_max_source_timestamp_utc,
        target.timestamp_utc as target_timestamp_utc,
        target.load_mw as target_load_mw,
        target.pool_price_cad_mwh as actual_pool_price_cad_mwh,
        target.hour,
        target.day_of_week,
        target.month,
        target.is_weekend,
        target.is_alberta_stat_holiday,
        target.is_canada_stat_holiday,
        target.is_stat_holiday,
        target.is_long_weekend,
        target.is_non_workday,
        target.is_workday,
        case
            when horizons.horizon_hours <= 1 then target.load_lag_1h
            else null
        end as load_lag_1h,
        case
            when horizons.horizon_hours <= 24 then target.load_lag_24h
            else null
        end as load_lag_24h,
        case
            when horizons.horizon_hours <= 168 then target.load_lag_168h
            else null
        end as load_lag_168h,
        issue.load_rolling_24h_avg,
        issue.load_rolling_168h_avg,
        case
            when horizons.horizon_hours <= 24 then target.price_lag_24h
            else null
        end as price_lag_24h,
        issue.price_rolling_24h_avg,
        issue.pool_price_cad_mwh as latest_known_pool_price_cad_mwh,
        weather.forecast_issue_utc as weather_forecast_issue_utc,
        weather.forecast_target_utc as weather_forecast_target_utc,
        case
            when weather.forecast_issue_utc is null then null
            else extract(epoch from (issue.timestamp_utc - weather.forecast_issue_utc)) / 3600
        end::numeric(10, 2) as weather_forecast_age_hours,
        weather.weather_forecast_region_count,
        weather.forecast_temperature_c,
        weather.forecast_wind_speed_mps,
        weather.forecast_relative_humidity_pct,
        weather.forecast_precipitation_mm,
        weather.forecast_heating_degree_c,
        weather.forecast_cooling_degree_c,
        generation.availability_issue_utc as generation_availability_issue_utc,
        generation.availability_target_utc as generation_availability_target_utc,
        case
            when generation.availability_issue_utc is null then null
            else extract(epoch from (issue.timestamp_utc - generation.availability_issue_utc)) / 3600
        end::numeric(10, 2) as generation_availability_age_hours,
        generation.generation_availability_fuel_type_count,
        generation.available_capacity_total_mw,
        generation.outage_capacity_total_mw,
        generation.derated_capacity_total_mw,
        generation.available_capacity_gas_mw,
        generation.available_capacity_coal_mw,
        generation.available_capacity_hydro_mw,
        generation.available_capacity_wind_mw,
        generation.available_capacity_solar_mw,
        generation.available_capacity_storage_mw,
        generation.available_capacity_other_mw,
        intertie.schedule_issue_utc as intertie_schedule_issue_utc,
        intertie.schedule_target_utc as intertie_schedule_target_utc,
        case
            when intertie.schedule_issue_utc is null then null
            else extract(epoch from (issue.timestamp_utc - intertie.schedule_issue_utc)) / 3600
        end::numeric(10, 2) as intertie_schedule_age_hours,
        intertie.intertie_count,
        intertie.scheduled_import_mw,
        intertie.scheduled_export_mw,
        intertie.net_scheduled_import_mw,
        intertie.import_transfer_limit_mw,
        intertie.export_transfer_limit_mw,
        intertie.intertie_constraint_mw,
        intertie.import_headroom_mw,
        intertie.export_headroom_mw,
        target.latest_ingestion_time
    from target_hours as target
    cross join horizons
    inner join {{ ref('int_energy_features_hourly') }} as issue
        on issue.timestamp_utc = target.timestamp_utc - (horizons.horizon_hours * interval '1 hour')
    left join lateral (
        select *
        from {{ ref('int_weather_forecast_features_hourly') }} as weather_features
        where weather_features.forecast_target_utc = target.timestamp_utc
            and weather_features.forecast_issue_utc <= issue.timestamp_utc
        order by weather_features.forecast_issue_utc desc
        limit 1
    ) as weather on true
    left join lateral (
        select *
        from {{ ref('int_generation_availability_features_hourly') }} as generation_features
        where generation_features.availability_target_utc = target.timestamp_utc
            and generation_features.availability_issue_utc <= issue.timestamp_utc
        order by generation_features.availability_issue_utc desc
        limit 1
    ) as generation on true
    left join lateral (
        select *
        from {{ ref('int_intertie_features_hourly') }} as intertie_features
        where intertie_features.schedule_target_utc = target.timestamp_utc
            and intertie_features.schedule_issue_utc <= issue.timestamp_utc
        order by intertie_features.schedule_issue_utc desc
        limit 1
    ) as intertie on true
    where issue.load_mw is not null
)

select *
from training_rows
