with horizons as (
    select horizon_hours
    from {{ ref('mart_forecast_training_set') }}
    group by horizon_hours
)

select expected.horizon_hours
from (values (1), (24)) as expected(horizon_hours)
left join horizons
    on expected.horizon_hours = horizons.horizon_hours
where horizons.horizon_hours is null
