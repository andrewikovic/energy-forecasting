with source_rows as (
    select
        interval_start_utc::timestamptz as timestamp_utc,
        alberta_internal_load_mw::numeric(12, 2) as load_mw,
        source::text as source,
        ingested_at::timestamptz as ingested_at
    from {{ source('raw', 'raw_aeso_load') }}
    where interval_start_utc is not null
),

deduped as (
    select
        *,
        row_number() over (
            partition by timestamp_utc
            order by ingested_at desc nulls last
        ) as row_number
    from source_rows
)

select
    timestamp_utc,
    load_mw,
    source,
    ingested_at
from deduped
where row_number = 1
