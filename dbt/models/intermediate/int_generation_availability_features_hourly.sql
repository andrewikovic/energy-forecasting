select
    availability_issue_utc,
    availability_target_utc,
    availability_horizon_hours,
    count(distinct fuel_type) as generation_availability_fuel_type_count,
    sum(available_capacity_mw)::numeric(14, 2) as available_capacity_total_mw,
    sum(outage_capacity_mw)::numeric(14, 2) as outage_capacity_total_mw,
    sum(derated_capacity_mw)::numeric(14, 2) as derated_capacity_total_mw,
    sum(available_capacity_mw) filter (
        where fuel_type in ('gas', 'natural_gas', 'natural gas')
    )::numeric(14, 2) as available_capacity_gas_mw,
    sum(available_capacity_mw) filter (
        where fuel_type = 'coal'
    )::numeric(14, 2) as available_capacity_coal_mw,
    sum(available_capacity_mw) filter (
        where fuel_type = 'hydro'
    )::numeric(14, 2) as available_capacity_hydro_mw,
    sum(available_capacity_mw) filter (
        where fuel_type = 'wind'
    )::numeric(14, 2) as available_capacity_wind_mw,
    sum(available_capacity_mw) filter (
        where fuel_type = 'solar'
    )::numeric(14, 2) as available_capacity_solar_mw,
    sum(available_capacity_mw) filter (
        where fuel_type = 'storage'
    )::numeric(14, 2) as available_capacity_storage_mw,
    sum(available_capacity_mw) filter (
        where fuel_type not in (
            'gas',
            'natural_gas',
            'natural gas',
            'coal',
            'hydro',
            'wind',
            'solar',
            'storage'
        )
    )::numeric(14, 2) as available_capacity_other_mw,
    max(ingested_at) as latest_generation_availability_ingested_at
from {{ ref('stg_generation_availability') }}
group by
    availability_issue_utc,
    availability_target_utc,
    availability_horizon_hours
