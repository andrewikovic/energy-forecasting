with aggregated as (
    select
        schedule_issue_utc,
        schedule_target_utc,
        schedule_horizon_hours,
        count(distinct intertie_id) as intertie_count,
        sum(scheduled_import_mw)::numeric(14, 2) as scheduled_import_mw,
        sum(scheduled_export_mw)::numeric(14, 2) as scheduled_export_mw,
        sum(transfer_limit_import_mw)::numeric(14, 2) as import_transfer_limit_mw,
        sum(transfer_limit_export_mw)::numeric(14, 2) as export_transfer_limit_mw,
        sum(constraint_mw)::numeric(14, 2) as intertie_constraint_mw,
        max(ingested_at) as latest_intertie_schedule_ingested_at
    from {{ ref('stg_intertie_schedules') }}
    group by
        schedule_issue_utc,
        schedule_target_utc,
        schedule_horizon_hours
)

select
    schedule_issue_utc,
    schedule_target_utc,
    schedule_horizon_hours,
    intertie_count,
    scheduled_import_mw,
    scheduled_export_mw,
    (scheduled_import_mw - scheduled_export_mw)::numeric(14, 2) as net_scheduled_import_mw,
    import_transfer_limit_mw,
    export_transfer_limit_mw,
    intertie_constraint_mw,
    (import_transfer_limit_mw - scheduled_import_mw)::numeric(14, 2) as import_headroom_mw,
    (export_transfer_limit_mw - scheduled_export_mw)::numeric(14, 2) as export_headroom_mw,
    latest_intertie_schedule_ingested_at
from aggregated
