select *
from {{ ref('mart_forecast_training_set') }}
where feature_max_source_timestamp_utc > feature_as_of_timestamp_utc
