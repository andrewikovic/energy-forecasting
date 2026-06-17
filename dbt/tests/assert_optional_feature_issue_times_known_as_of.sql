select *
from {{ ref('mart_forecast_training_set') }}
where (
    weather_forecast_issue_utc is not null
    and weather_forecast_issue_utc > feature_as_of_timestamp_utc
)
or (
    generation_availability_issue_utc is not null
    and generation_availability_issue_utc > feature_as_of_timestamp_utc
)
or (
    intertie_schedule_issue_utc is not null
    and intertie_schedule_issue_utc > feature_as_of_timestamp_utc
)
