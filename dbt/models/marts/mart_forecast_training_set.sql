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
        issue.timestamp_utc as feature_max_source_timestamp_utc,
        target.timestamp_utc as target_timestamp_utc,
        target.load_mw as target_load_mw,
        target.pool_price_cad_mwh as actual_pool_price_cad_mwh,
        target.hour,
        target.day_of_week,
        target.month,
        target.is_weekend,
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
        target.latest_ingestion_time
    from target_hours as target
    cross join horizons
    inner join {{ ref('int_energy_features_hourly') }} as issue
        on issue.timestamp_utc = target.timestamp_utc - (horizons.horizon_hours * interval '1 hour')
    where issue.load_mw is not null
)

select *
from training_rows
