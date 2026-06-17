{% set raw_generation_relation = adapter.get_relation(
    database=target.database,
    schema='raw',
    identifier='raw_generation_availability_hourly'
) %}

{% if not var('enable_generation_availability_features', true) or raw_generation_relation is none %}

select
    cast(null as timestamptz) as availability_issue_utc,
    cast(null as timestamptz) as availability_target_utc,
    cast(null as integer) as availability_horizon_hours,
    cast(null as text) as fuel_type,
    cast(null as text) as unit_id,
    cast(null as numeric(12, 2)) as available_capacity_mw,
    cast(null as numeric(12, 2)) as derated_capacity_mw,
    cast(null as numeric(12, 2)) as outage_capacity_mw,
    cast(null as text) as source,
    cast(null as timestamptz) as ingested_at
where false

{% else %}

with source_rows as (
    select
        availability_issue_utc::timestamptz as availability_issue_utc,
        availability_target_utc::timestamptz as availability_target_utc,
        lower(coalesce(nullif(trim(fuel_type::text), ''), 'unknown')) as fuel_type,
        coalesce(nullif(trim(unit_id::text), ''), 'unreported') as unit_id,
        available_capacity_mw::numeric(12, 2) as available_capacity_mw,
        derated_capacity_mw::numeric(12, 2) as derated_capacity_mw,
        outage_capacity_mw::numeric(12, 2) as outage_capacity_mw,
        source::text as source,
        ingested_at::timestamptz as ingested_at
    from {{ raw_generation_relation }}
    where availability_issue_utc is not null
        and availability_target_utc is not null
        and availability_target_utc >= availability_issue_utc
),

deduped as (
    select
        *,
        row_number() over (
            partition by availability_issue_utc, availability_target_utc, fuel_type, unit_id
            order by ingested_at desc nulls last
        ) as row_number
    from source_rows
)

select
    availability_issue_utc,
    availability_target_utc,
    (extract(epoch from (availability_target_utc - availability_issue_utc)) / 3600)::integer
        as availability_horizon_hours,
    fuel_type,
    unit_id,
    available_capacity_mw,
    derated_capacity_mw,
    outage_capacity_mw,
    source,
    ingested_at
from deduped
where row_number = 1

{% endif %}
