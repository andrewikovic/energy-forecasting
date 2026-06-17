select *
from {{ ref('mart_forecast_training_set') }}
where (
    weather_forecast_target_utc is not null
    and weather_forecast_target_utc != target_timestamp_utc
)
or (
    generation_availability_target_utc is not null
    and generation_availability_target_utc != target_timestamp_utc
)
or (
    intertie_schedule_target_utc is not null
    and intertie_schedule_target_utc != target_timestamp_utc
)
