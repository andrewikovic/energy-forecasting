with weather_duplicates as (
    select
        'int_weather_forecast_features_hourly' as model_name,
        forecast_issue_utc as issue_utc,
        forecast_target_utc as target_utc,
        count(*) as row_count
    from {{ ref('int_weather_forecast_features_hourly') }}
    group by forecast_issue_utc, forecast_target_utc
    having count(*) > 1
),

generation_duplicates as (
    select
        'int_generation_availability_features_hourly' as model_name,
        availability_issue_utc as issue_utc,
        availability_target_utc as target_utc,
        count(*) as row_count
    from {{ ref('int_generation_availability_features_hourly') }}
    group by availability_issue_utc, availability_target_utc
    having count(*) > 1
),

intertie_duplicates as (
    select
        'int_intertie_features_hourly' as model_name,
        schedule_issue_utc as issue_utc,
        schedule_target_utc as target_utc,
        count(*) as row_count
    from {{ ref('int_intertie_features_hourly') }}
    group by schedule_issue_utc, schedule_target_utc
    having count(*) > 1
)

select * from weather_duplicates
union all
select * from generation_duplicates
union all
select * from intertie_duplicates
