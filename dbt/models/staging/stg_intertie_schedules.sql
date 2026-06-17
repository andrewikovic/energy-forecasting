{% set raw_intertie_relation = adapter.get_relation(
    database=target.database,
    schema='raw',
    identifier='raw_intertie_schedule_hourly'
) %}

{% if not var('enable_intertie_features', true) or raw_intertie_relation is none %}

select
    cast(null as timestamptz) as schedule_issue_utc,
    cast(null as timestamptz) as schedule_target_utc,
    cast(null as integer) as schedule_horizon_hours,
    cast(null as text) as intertie_id,
    cast(null as numeric(12, 2)) as scheduled_import_mw,
    cast(null as numeric(12, 2)) as scheduled_export_mw,
    cast(null as numeric(12, 2)) as transfer_limit_import_mw,
    cast(null as numeric(12, 2)) as transfer_limit_export_mw,
    cast(null as numeric(12, 2)) as constraint_mw,
    cast(null as text) as source,
    cast(null as timestamptz) as ingested_at
where false

{% else %}

with source_rows as (
    select
        schedule_issue_utc::timestamptz as schedule_issue_utc,
        schedule_target_utc::timestamptz as schedule_target_utc,
        lower(coalesce(nullif(trim(intertie_id::text), ''), 'unknown')) as intertie_id,
        scheduled_import_mw::numeric(12, 2) as scheduled_import_mw,
        scheduled_export_mw::numeric(12, 2) as scheduled_export_mw,
        transfer_limit_import_mw::numeric(12, 2) as transfer_limit_import_mw,
        transfer_limit_export_mw::numeric(12, 2) as transfer_limit_export_mw,
        constraint_mw::numeric(12, 2) as constraint_mw,
        source::text as source,
        ingested_at::timestamptz as ingested_at
    from {{ raw_intertie_relation }}
    where schedule_issue_utc is not null
        and schedule_target_utc is not null
        and schedule_target_utc >= schedule_issue_utc
),

deduped as (
    select
        *,
        row_number() over (
            partition by schedule_issue_utc, schedule_target_utc, intertie_id
            order by ingested_at desc nulls last
        ) as row_number
    from source_rows
)

select
    schedule_issue_utc,
    schedule_target_utc,
    (extract(epoch from (schedule_target_utc - schedule_issue_utc)) / 3600)::integer
        as schedule_horizon_hours,
    intertie_id,
    scheduled_import_mw,
    scheduled_export_mw,
    transfer_limit_import_mw,
    transfer_limit_export_mw,
    constraint_mw,
    source,
    ingested_at
from deduped
where row_number = 1

{% endif %}
