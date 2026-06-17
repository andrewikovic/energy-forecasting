from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from aeso_analytics.database import get_engine


st.set_page_config(page_title="AESO Load Forecasting", layout="wide")


@st.cache_data(ttl=120)
def query(sql: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql_query(text(sql), engine)


def load_table(sql: str, missing_message: str) -> pd.DataFrame | None:
    try:
        return query(sql)
    except SQLAlchemyError:
        st.warning(missing_message)
        return None


def metric_row(metrics: dict[str, str | int | float]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, value)


def market_overview() -> None:
    df = load_table(
        "select * from marts.mart_market_overview order by timestamp_utc",
        "Market mart is not available. Run `make ingest` and `make dbt-run`.",
    )
    if df is None or df.empty:
        return

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    latest = df.iloc[-1]
    metric_row(
        {
            "Latest load MW": f"{latest['load_mw']:,.0f}",
            "Average load MW": f"{df['load_mw'].mean():,.0f}",
            "Peak load MW": f"{df['load_mw'].max():,.0f}",
            "Average pool price": f"${df['pool_price_cad_mwh'].mean():,.2f}",
            "Hourly records": f"{len(df):,}",
        }
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(df, x="timestamp_utc", y="load_mw", title="Hourly Alberta Internal Load"),
            width="stretch",
        )
        daily = (
            df.groupby("date", as_index=False)
            .agg(daily_average_load_mw=("load_mw", "mean"), daily_peak_load_mw=("load_mw", "max"))
        )
        st.plotly_chart(
            px.line(
                daily,
                x="date",
                y=["daily_average_load_mw", "daily_peak_load_mw"],
                title="Daily Average and Peak Load",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.line(df, x="timestamp_utc", y="pool_price_cad_mwh", title="Pool Price Over Time"),
            width="stretch",
        )
        st.plotly_chart(
            px.scatter(
                df,
                x="load_mw",
                y="pool_price_cad_mwh",
                color="is_weekend",
                title="Load vs Pool Price",
                opacity=0.65,
            ),
            width="stretch",
        )


def forecast_results() -> pd.DataFrame | None:
    df = load_table(
        """
        select *
        from ml.forecast_results
        order by target_timestamp_utc
        """,
        "Forecast results are not available. Run `make train` after dbt has built the marts.",
    )
    if df is None or df.empty:
        return df
    df["target_timestamp_utc"] = pd.to_datetime(df["target_timestamp_utc"], utc=True)
    return df


def filtered_forecasts(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    horizons = sorted(df["horizon_hours"].unique())
    models = sorted(df["model_name"].unique())
    splits = sorted(df["evaluation_split"].unique())
    left, middle, right = st.columns(3)
    horizon = left.selectbox("Horizon", horizons, key=f"{key_prefix}_horizon")
    model = middle.selectbox("Model", models, key=f"{key_prefix}_model")
    split = right.selectbox(
        "Split",
        splits,
        index=splits.index("test") if "test" in splits else 0,
        key=f"{key_prefix}_split",
    )
    return df[
        (df["horizon_hours"] == horizon)
        & (df["model_name"] == model)
        & (df["evaluation_split"] == split)
    ].copy()


def load_forecast() -> None:
    df = forecast_results()
    if df is None or df.empty:
        return
    selected = filtered_forecasts(df, "forecast")
    if selected.empty:
        st.warning("No forecast rows match the selected filters.")
        return

    metric_row(
        {
            "MAE MW": f"{selected['absolute_error_mw'].mean():,.1f}",
            "RMSE MW": f"{(selected['error_mw'].pow(2).mean() ** 0.5):,.1f}",
            "MAPE": f"{selected['absolute_percentage_error'].mean():,.2f}%",
            "Underprediction": f"{(selected['error_mw'] < 0).mean() * 100:,.1f}%",
        }
    )

    st.plotly_chart(
        px.line(
            selected,
            x="target_timestamp_utc",
            y=["target_load_mw", "predicted_load_mw"],
            title="Actual vs Predicted Load",
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(selected, x="target_timestamp_utc", y="error_mw", title="Forecast Error Over Time"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.histogram(selected, x="error_mw", nbins=40, title="Residual Distribution"),
            width="stretch",
        )

    st.dataframe(
        selected[
            [
                "target_timestamp_utc",
                "target_load_mw",
                "predicted_load_mw",
                "error_mw",
                "absolute_percentage_error",
                "is_peak_period",
            ]
        ].sort_values("target_timestamp_utc", ascending=False),
        width="stretch",
        hide_index=True,
    )


def model_comparison() -> None:
    performance = load_table(
        """
        select *
        from ml.model_performance
        order by horizon_hours, evaluation_split, mae
        """,
        "Model performance is not available. Run `make train`.",
    )
    if performance is None or performance.empty:
        return

    split_options = sorted(performance["evaluation_split"].unique())
    split = st.selectbox(
        "Split",
        split_options,
        index=split_options.index("test") if "test" in split_options else 0,
    )
    selected = performance[performance["evaluation_split"] == split].copy()

    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(
            px.bar(selected, x="model_name", y="mae", color="horizon_hours", barmode="group", title="MAE"),
            width="stretch",
        )
    with middle:
        st.plotly_chart(
            px.bar(selected, x="model_name", y="rmse", color="horizon_hours", barmode="group", title="RMSE"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.bar(selected, x="model_name", y="mape", color="horizon_hours", barmode="group", title="MAPE"),
            width="stretch",
        )

    st.dataframe(selected.sort_values(["horizon_hours", "mae"]), width="stretch", hide_index=True)


def error_analysis() -> None:
    df = forecast_results()
    if df is None or df.empty:
        return
    selected = filtered_forecasts(df, "errors")
    selected["hour"] = selected["target_timestamp_utc"].dt.hour
    selected["day_of_week"] = selected["target_timestamp_utc"].dt.day_name()
    selected["month"] = selected["target_timestamp_utc"].dt.month

    by_hour = selected.groupby("hour", as_index=False).agg(mae=("absolute_error_mw", "mean"))
    by_day = selected.groupby("day_of_week", as_index=False).agg(mae=("absolute_error_mw", "mean"))
    by_month = selected.groupby("month", as_index=False).agg(mae=("absolute_error_mw", "mean"))

    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(px.bar(by_hour, x="hour", y="mae", title="Error by Hour"), width="stretch")
    with middle:
        st.plotly_chart(
            px.bar(by_day, x="day_of_week", y="mae", title="Error by Day of Week"),
            width="stretch",
        )
    with right:
        st.plotly_chart(px.bar(by_month, x="month", y="mae", title="Error by Month"), width="stretch")

    peaks = selected[selected["is_peak_period"]].copy()
    metric_row(
        {
            "Peak MAE MW": f"{peaks['absolute_error_mw'].mean():,.1f}" if not peaks.empty else "n/a",
            "Peak underprediction": f"{(peaks['error_mw'] < 0).mean() * 100:,.1f}%" if not peaks.empty else "n/a",
            "Peak rows": f"{len(peaks):,}",
        }
    )
    st.dataframe(
        peaks.sort_values("absolute_error_mw", ascending=False).head(50),
        width="stretch",
        hide_index=True,
    )


def peak_demand_monitor() -> None:
    peaks = load_table(
        "select * from marts.mart_peak_demand_events order by peak_rank limit 200",
        "Peak demand mart is not available. Run `make dbt-run`.",
    )
    forecasts = forecast_results()
    if peaks is None or peaks.empty:
        return

    metric_row(
        {
            "Top load MW": f"{peaks['load_mw'].max():,.0f}",
            "Peak monitor rows": f"{len(peaks):,}",
            "Average peak price": f"${peaks['pool_price_cad_mwh'].mean():,.2f}",
        }
    )

    st.plotly_chart(
        px.bar(
            peaks.head(30).sort_values("peak_rank", ascending=False),
            x="load_mw",
            y="timestamp_utc",
            orientation="h",
            title="Top Load Hours",
        ),
        width="stretch",
    )

    if forecasts is None or forecasts.empty:
        return
    selected = filtered_forecasts(forecasts, "peak")
    selected_peaks = selected[selected["is_peak_period"]].copy()
    if selected_peaks.empty:
        st.warning("No peak-period forecast rows match the selected filters.")
        return

    metric_row(
        {
            "Selected peak MAE MW": f"{selected_peaks['absolute_error_mw'].mean():,.1f}",
            "Selected peak underprediction": f"{(selected_peaks['error_mw'] < 0).mean() * 100:,.1f}%",
            "Selected peak rows": f"{len(selected_peaks):,}",
        }
    )
    st.plotly_chart(
        px.scatter(
            selected_peaks,
            x="target_load_mw",
            y="predicted_load_mw",
            color="error_mw",
            title="Actual vs Predicted During Peak Periods",
        ),
        width="stretch",
    )


def data_quality() -> None:
    checks = load_table(
        """
        with load_stats as (
            select
                'raw.raw_aeso_load' as table_name,
                count(*) as record_count,
                min(interval_start_utc) as min_timestamp_utc,
                max(interval_start_utc) as max_timestamp_utc,
                max(ingested_at) as latest_ingestion_time,
                count(*) - count(alberta_internal_load_mw) as null_value_count,
                count(*) - count(distinct interval_start_utc) as duplicate_timestamp_count
            from raw.raw_aeso_load
        ),
        price_stats as (
            select
                'raw.raw_aeso_pool_price' as table_name,
                count(*) as record_count,
                min(interval_start_utc) as min_timestamp_utc,
                max(interval_start_utc) as max_timestamp_utc,
                max(ingested_at) as latest_ingestion_time,
                count(*) - count(pool_price_cad_mwh) as null_value_count,
                count(*) - count(distinct interval_start_utc) as duplicate_timestamp_count
            from raw.raw_aeso_pool_price
        )
        select * from load_stats
        union all
        select * from price_stats
        """,
        "Raw tables are not available. Run `make ingest`.",
    )
    if checks is None or checks.empty:
        return

    missing_hours = load_table(
        """
        with bounds as (
            select min(interval_start_utc) as min_ts, max(interval_start_utc) as max_ts
            from raw.raw_aeso_load
        ),
        spine as (
            select generate_series(min_ts, max_ts, interval '1 hour') as timestamp_utc
            from bounds
        )
        select count(*) as missing_hour_count
        from spine
        left join raw.raw_aeso_load
            on spine.timestamp_utc = raw.raw_aeso_load.interval_start_utc
        where raw.raw_aeso_load.interval_start_utc is null
        """,
        "Could not calculate missing hours.",
    )

    metric_row(
        {
            "Raw load rows": f"{int(checks.loc[checks['table_name'] == 'raw.raw_aeso_load', 'record_count'].iloc[0]):,}",
            "Raw price rows": f"{int(checks.loc[checks['table_name'] == 'raw.raw_aeso_pool_price', 'record_count'].iloc[0]):,}",
            "Missing load hours": f"{int(missing_hours['missing_hour_count'].iloc[0]):,}" if missing_hours is not None else "n/a",
        }
    )
    st.dataframe(checks, width="stretch", hide_index=True)


PAGES = {
    "Market Overview": market_overview,
    "Load Forecast": load_forecast,
    "Model Comparison": model_comparison,
    "Error Analysis": error_analysis,
    "Peak Demand Monitor": peak_demand_monitor,
    "Data Quality": data_quality,
}


st.title("AESO Load Forecasting & Market Analytics")
page = st.sidebar.radio("Page", list(PAGES))
PAGES[page]()
