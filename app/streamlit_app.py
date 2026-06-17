from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from aeso_analytics.dashboard_governance import (
    HOLIDAY_FLAG_COLUMNS,
    available_accuracy_metrics,
    current_champions,
    tag_feature_importance,
)
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


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


@st.cache_data(ttl=120)
def table_columns(schema: str, table: str) -> tuple[str, ...] | None:
    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table(table, schema=schema):
        return None
    return tuple(column["name"] for column in inspector.get_columns(table, schema=schema))


def load_optional_table(
    schema: str,
    table: str,
    missing_message: str,
    required_columns: Sequence[str] = (),
    order_by: Sequence[str] = (),
) -> pd.DataFrame | None:
    try:
        columns = table_columns(schema, table)
    except SQLAlchemyError:
        st.warning(missing_message)
        return None

    qualified_name = f"{schema}.{table}"
    if columns is None:
        st.info(f"{qualified_name} is not available. {missing_message}")
        return None

    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        st.info(
            f"{qualified_name} is missing required columns: "
            f"{', '.join(missing_columns)}. {missing_message}"
        )
        return None

    order_columns = [column for column in order_by if column in columns]
    order_clause = (
        " order by " + ", ".join(quote_identifier(column) for column in order_columns)
        if order_columns
        else ""
    )
    sql = (
        f"select * from {quote_identifier(schema)}.{quote_identifier(table)}"
        f"{order_clause}"
    )
    return load_table(sql, missing_message)


def latest_run_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "run_id" not in df.columns:
        return []

    runs = df[["run_id"]].dropna().drop_duplicates().copy()
    runs["run_id"] = runs["run_id"].astype(str)
    if "evaluated_at" in df.columns:
        evaluated = df[["run_id", "evaluated_at"]].dropna(subset=["run_id"]).copy()
        evaluated["run_id"] = evaluated["run_id"].astype(str)
        evaluated["evaluated_at"] = pd.to_datetime(
            evaluated["evaluated_at"],
            utc=True,
            errors="coerce",
        )
        runs = runs.merge(
            evaluated.groupby("run_id", as_index=False).agg(evaluated_at=("evaluated_at", "max")),
            on="run_id",
            how="left",
        ).sort_values("evaluated_at", ascending=False, na_position="last")
    else:
        runs = runs.sort_values("run_id", ascending=False)
    return runs["run_id"].tolist()


def filter_by_run(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    options = latest_run_options(df)
    if not options:
        return df
    selected_run = st.selectbox(label, options, key=key)
    return df[df["run_id"].astype(str) == selected_run].copy()


def filter_by_horizon(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty or "horizon_hours" not in df.columns:
        return df
    horizons = sorted(df["horizon_hours"].dropna().unique())
    if not horizons:
        return df
    selected = st.selectbox("Horizon", horizons, key=key)
    return df[df["horizon_hours"] == selected].copy()


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


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


def governance_registry() -> pd.DataFrame | None:
    return load_optional_table(
        "ml",
        "model_registry",
        "Run `make train` to populate the model registry.",
        order_by=["horizon_hours", "is_preferred", "preferred_at", "trained_at"],
    )


def governance_backtests() -> pd.DataFrame | None:
    return load_optional_table(
        "ml",
        "model_backtest_results",
        "Run `make train` with backtesting enabled, then `make evaluate`.",
        order_by=["evaluated_at", "horizon_hours", "test_start_timestamp_utc", "model_name"],
    )


def governance_decisions() -> pd.DataFrame | None:
    return load_optional_table(
        "ml",
        "model_promotion_decisions",
        "Run `make train` with backtesting enabled, then `make evaluate`.",
        order_by=["evaluated_at", "horizon_hours", "decision"],
    )


def governance_feature_importance() -> pd.DataFrame | None:
    return load_optional_table(
        "ml",
        "feature_importance",
        "Run `make train` to populate feature importance.",
        order_by=["created_at", "horizon_hours", "importance"],
    )


def current_champion_section(registry: pd.DataFrame | None) -> None:
    st.subheader("Current Champion")
    if registry is None or registry.empty:
        st.info("No model registry rows are available.")
        return
    if "is_preferred" not in registry.columns:
        st.info("ml.model_registry does not include is_preferred, so the current champion cannot be identified.")
        return

    champions = current_champions(registry)
    if champions.empty:
        st.info("No preferred champion is currently marked in ml.model_registry.")
        return

    for column in ("preferred_at", "trained_at", "created_at"):
        if column in champions.columns:
            champions[column] = pd.to_datetime(champions[column], utc=True, errors="coerce")

    latest_timestamp = None
    for column in ("preferred_at", "trained_at", "created_at"):
        if column in champions.columns and champions[column].notna().any():
            latest_timestamp = champions[column].max()
            break

    metric_row(
        {
            "Preferred horizons": f"{len(champions):,}",
            "Champion models": f"{champions['model_name'].nunique():,}" if "model_name" in champions.columns else "n/a",
            "Latest champion date": latest_timestamp.strftime("%Y-%m-%d") if latest_timestamp is not None else "n/a",
        }
    )

    display_columns = available_columns(
        champions,
        [
            "horizon_hours",
            "model_name",
            "model_id",
            "model_version",
            "version",
            "model_type",
            "run_id",
            "is_preferred",
            "preferred_at",
            "trained_at",
            "created_at",
            "train_start_timestamp_utc",
            "train_end_timestamp_utc",
            "artifact_path",
            "promotion_decision_id",
        ],
    )
    display = champions[display_columns].copy()
    rename_map = {
        "model_id": "model_version",
        "trained_at": "created_at",
    }
    display = display.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in display.columns and new not in display.columns
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def backtest_window_section(backtests: pd.DataFrame | None, registry: pd.DataFrame | None) -> None:
    st.subheader("Backtest Windows")
    if backtests is None or backtests.empty:
        st.info("No backtest rows are available in ml.model_backtest_results.")
        return

    metrics = available_accuracy_metrics(backtests)
    if not metrics:
        st.info("ml.model_backtest_results does not include MAE, RMSE, or MAPE columns.")
        return

    selected = filter_by_run(backtests, "governance_backtest_run", "Backtest run")
    selected = filter_by_horizon(selected, "governance_backtest_horizon")
    if selected.empty:
        st.info("No backtest rows match the selected filters.")
        return

    metric = st.selectbox(
        "Metric",
        metrics,
        format_func=lambda value: value.upper(),
        key="governance_backtest_metric",
    )

    for column in ("test_start_timestamp_utc", "test_end_timestamp_utc", "evaluated_at"):
        if column in selected.columns:
            selected[column] = pd.to_datetime(selected[column], utc=True, errors="coerce")

    if registry is not None and not registry.empty and "model_id" in selected.columns:
        champion_ids = set()
        champions = current_champions(registry)
        if "model_id" in champions.columns:
            if "horizon_hours" in selected.columns and "horizon_hours" in champions.columns:
                champion_ids = set(
                    champions[
                        champions["horizon_hours"].isin(selected["horizon_hours"].dropna().unique())
                    ]["model_id"].astype(str)
                )
            else:
                champion_ids = set(champions["model_id"].astype(str))
        if champion_ids:
            selected["governance_role"] = selected["model_id"].astype(str).map(
                lambda value: "Champion" if value in champion_ids else "Challenger"
            )

    metric_row(
        {
            "Backtest windows": f"{selected['window_id'].nunique():,}" if "window_id" in selected.columns else f"{len(selected):,}",
            "Models evaluated": f"{selected['model_name'].nunique():,}" if "model_name" in selected.columns else "n/a",
            f"Average {metric.upper()}": f"{pd.to_numeric(selected[metric], errors='coerce').mean():,.2f}",
        }
    )

    chart_frame = selected.copy()
    x_column = "test_start_timestamp_utc" if "test_start_timestamp_utc" in chart_frame.columns else "window_id"
    if x_column in chart_frame.columns and "model_name" in chart_frame.columns:
        hover_columns = available_columns(
            chart_frame,
            [
                "window_id",
                "test_start_timestamp_utc",
                "test_end_timestamp_utc",
                "train_start_timestamp_utc",
                "train_end_timestamp_utc",
                "governance_role",
            ],
        )
        st.plotly_chart(
            px.line(
                chart_frame.sort_values(x_column),
                x=x_column,
                y=metric,
                color="model_name",
                line_dash="governance_role" if "governance_role" in chart_frame.columns else None,
                markers=True,
                hover_data=hover_columns,
                title=f"{metric.upper()} by Backtest Window",
            ),
            width="stretch",
        )
    else:
        st.info("Backtest window chart needs model_name plus window_id or test_start_timestamp_utc.")

    if "model_name" in selected.columns:
        summary_group = [
            column
            for column in ("horizon_hours", "model_name", "governance_role")
            if column in selected.columns
        ]
        summary = (
            selected.assign(**{metric: pd.to_numeric(selected[metric], errors="coerce")})
            .groupby(summary_group, dropna=False)
            .agg(average_metric=(metric, "mean"), valid_windows=(metric, "count"))
            .reset_index()
            .sort_values("average_metric")
        )
        st.plotly_chart(
            px.bar(
                summary,
                x="model_name",
                y="average_metric",
                color="governance_role" if "governance_role" in summary.columns else "model_name",
                title=f"Average {metric.upper()} by Model",
                hover_data=available_columns(summary, ["horizon_hours", "valid_windows"]),
            ),
            width="stretch",
        )

    detail_columns = available_columns(
        selected,
        [
            "run_id",
            "horizon_hours",
            "model_name",
            "model_type",
            "governance_role",
            "window_id",
            "backtest_mode",
            "train_start_timestamp_utc",
            "train_end_timestamp_utc",
            "test_start_timestamp_utc",
            "test_end_timestamp_utc",
            "train_row_count",
            "test_row_count",
            "prediction_row_count",
            "mae",
            "rmse",
            "mape",
            "peak_period_mae",
            "underprediction_rate",
            "evaluated_at",
        ],
    )
    st.dataframe(selected[detail_columns], width="stretch", hide_index=True)


def promotion_decision_section(decisions: pd.DataFrame | None) -> None:
    st.subheader("Promotion Decisions")
    if decisions is None or decisions.empty:
        st.info("No promotion decision rows are available in ml.model_promotion_decisions.")
        return

    selected = filter_by_run(decisions, "governance_decision_run", "Decision run")
    left, middle = st.columns(2)
    if "horizon_hours" in selected.columns:
        horizons = sorted(selected["horizon_hours"].dropna().unique())
        chosen_horizons = left.multiselect(
            "Horizons",
            horizons,
            default=horizons,
            key="governance_decision_horizons",
        )
        selected = selected[selected["horizon_hours"].isin(chosen_horizons)]
    if "decision" in selected.columns:
        decisions_available = sorted(selected["decision"].dropna().unique())
        chosen_decisions = middle.multiselect(
            "Decisions",
            decisions_available,
            default=decisions_available,
            key="governance_decision_values",
        )
        selected = selected[selected["decision"].isin(chosen_decisions)]

    if selected.empty:
        st.info("No promotion decisions match the selected filters.")
        return

    promoted_count = int((selected["decision"] == "promoted").sum()) if "decision" in selected.columns else 0
    rejected_count = int((selected["decision"] == "rejected").sum()) if "decision" in selected.columns else 0
    latest = pd.to_datetime(selected["evaluated_at"], utc=True, errors="coerce").max() if "evaluated_at" in selected.columns else None
    metric_row(
        {
            "Promoted": f"{promoted_count:,}",
            "Rejected": f"{rejected_count:,}",
            "Latest decision": latest.strftime("%Y-%m-%d %H:%M UTC") if latest is not None and pd.notna(latest) else "n/a",
        }
    )

    columns = available_columns(
        selected,
        [
            "evaluated_at",
            "run_id",
            "horizon_hours",
            "champion_model_name",
            "challenger_model_name",
            "decision",
            "reason",
            "valid_window_count",
            "required_window_count",
            "win_count",
            "win_rate",
            "champion_average_mae",
            "challenger_average_mae",
            "champion_average_rmse",
            "challenger_average_rmse",
            "champion_average_peak_period_mae",
            "challenger_average_peak_period_mae",
            "champion_average_underprediction_rate",
            "challenger_average_underprediction_rate",
            "promoted_model_id",
            "rules",
            "comparison_details",
        ],
    )
    st.dataframe(selected[columns], width="stretch", hide_index=True)


def feature_importance_section(importance: pd.DataFrame | None) -> None:
    st.subheader("Feature Impact")
    if importance is None or importance.empty:
        st.info("No feature importance rows are available in ml.feature_importance.")
        return
    if "feature_name" not in importance.columns or "importance" not in importance.columns:
        st.info("ml.feature_importance needs feature_name and importance columns for this view.")
        return

    selected = filter_by_horizon(importance, "governance_importance_horizon")
    if "model_name" in selected.columns:
        models = sorted(selected["model_name"].dropna().unique())
        model = st.selectbox("Model", models, key="governance_importance_model")
        selected = selected[selected["model_name"] == model].copy()

    selected = tag_feature_importance(selected)
    selected["importance"] = pd.to_numeric(selected["importance"], errors="coerce")
    selected = selected.dropna(subset=["importance"])
    if selected.empty:
        st.info("No numeric feature importance values match the selected filters.")
        return

    top_n = st.slider(
        "Top features",
        min_value=5,
        max_value=min(30, max(5, len(selected))),
        value=min(15, max(5, len(selected))),
        key="governance_top_features",
    )
    top_features = selected.sort_values("importance", ascending=False).head(top_n)
    highlighted = top_features[top_features["feature_group"] != "Other"]
    metric_row(
        {
            "Features shown": f"{len(top_features):,}",
            "Weekly baseline signals": f"{(top_features['feature_group'] == 'Weekly baseline').sum():,}",
            "Holiday/calendar signals": f"{(top_features['feature_group'] == 'Holiday / calendar').sum():,}",
        }
    )

    st.plotly_chart(
        px.bar(
            top_features.sort_values("importance"),
            x="importance",
            y="feature_name",
            color="feature_group",
            orientation="h",
            title="Top Feature Importances",
        ),
        width="stretch",
    )
    if not highlighted.empty:
        st.dataframe(
            highlighted[available_columns(highlighted, ["horizon_hours", "model_name", "feature_name", "importance", "feature_group"])],
            width="stretch",
            hide_index=True,
        )


def precomputed_holiday_backtest_summary(backtests: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "horizon_hours",
        "model_name",
        "holiday_group",
        "mae",
        "row_count",
        "source",
    ]
    if backtests is None or backtests.empty:
        return pd.DataFrame(columns=columns)

    holiday_mae = next(
        (column for column in ("holiday_mae", "holiday_mean_absolute_error") if column in backtests.columns),
        None,
    )
    non_holiday_mae = next(
        (
            column
            for column in ("non_holiday_mae", "non_holiday_mean_absolute_error")
            if column in backtests.columns
        ),
        None,
    )
    if holiday_mae is None or non_holiday_mae is None:
        return pd.DataFrame(columns=columns)

    group_columns = available_columns(backtests, ["horizon_hours", "model_name"])
    summary = (
        backtests.melt(
            id_vars=group_columns,
            value_vars=[holiday_mae, non_holiday_mae],
            var_name="holiday_group",
            value_name="mae",
        )
        .assign(
            holiday_group=lambda frame: frame["holiday_group"].map(
                {
                    holiday_mae: "Holiday",
                    non_holiday_mae: "Non-holiday",
                }
            ),
            mae=lambda frame: pd.to_numeric(frame["mae"], errors="coerce"),
            source="ml.model_backtest_results",
        )
        .dropna(subset=["mae"])
    )
    summary = (
        summary.groupby([*group_columns, "holiday_group", "source"], dropna=False)
        .agg(mae=("mae", "mean"), row_count=("mae", "size"))
        .reset_index()
    )
    for column in columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary[columns]


def holiday_case_sql(alias: str, column: str) -> str:
    quoted = f"{alias}.{quote_identifier(column)}"
    return (
        "case when lower(cast("
        f"{quoted}"
        " as text)) in ('true', 't', '1', 'yes', 'y') "
        "then 'Holiday' else 'Non-holiday' end"
    )


def forecast_error_expression(columns: Sequence[str], alias: str = "f") -> str | None:
    if "absolute_error_mw" in columns:
        return f"{alias}.{quote_identifier('absolute_error_mw')}"
    if {"target_load_mw", "predicted_load_mw"}.issubset(columns):
        return (
            f"abs({alias}.{quote_identifier('predicted_load_mw')} - "
            f"{alias}.{quote_identifier('target_load_mw')})"
        )
    return None


def load_holiday_error_summary() -> pd.DataFrame | None:
    try:
        forecast_columns = table_columns("ml", "forecast_results")
    except SQLAlchemyError:
        st.warning("Could not inspect ml.forecast_results for holiday error comparison.")
        return None

    if forecast_columns is None:
        st.info("Holiday error comparison needs ml.forecast_results.")
        return None

    error_expression = forecast_error_expression(forecast_columns)
    if error_expression is None:
        st.info("Holiday error comparison needs absolute_error_mw or target/predicted load fields.")
        return None

    forecast_holiday_column = next(
        (column for column in HOLIDAY_FLAG_COLUMNS if column in forecast_columns),
        None,
    )
    group_columns = [
        column
        for column in ("horizon_hours", "model_name", "evaluation_split")
        if column in forecast_columns
    ]

    if forecast_holiday_column is not None:
        holiday_case = holiday_case_sql("f", forecast_holiday_column)
        select_columns = ", ".join(f"f.{quote_identifier(column)}" for column in group_columns)
        group_clause = ", ".join([*(f"f.{quote_identifier(column)}" for column in group_columns), holiday_case])
        sql = f"""
        select
            {select_columns + "," if select_columns else ""}
            {holiday_case} as holiday_group,
            avg({error_expression}) as mae,
            count(*) as row_count,
            'ml.forecast_results' as source
        from ml.forecast_results f
        where {error_expression} is not null
        group by {group_clause}
        order by {group_clause}
        """
        return load_table(sql, "Could not calculate holiday error comparison.")

    try:
        training_columns = table_columns("marts", "mart_forecast_training_set")
    except SQLAlchemyError:
        st.warning("Could not inspect marts.mart_forecast_training_set for holiday flags.")
        return None

    if training_columns is None:
        st.info("Holiday error comparison needs holiday flags in forecast results or the training mart.")
        return None

    training_holiday_column = next(
        (column for column in HOLIDAY_FLAG_COLUMNS if column in training_columns),
        None,
    )
    if training_holiday_column is None:
        st.info("No holiday flag columns are available for holiday vs non-holiday error comparison.")
        return None

    exact_join_columns = [
        column
        for column in ("horizon_hours", "feature_as_of_timestamp_utc", "target_timestamp_utc")
        if column in forecast_columns and column in training_columns
    ]
    fallback_join_columns = [
        column
        for column in ("horizon_hours", "target_timestamp_utc")
        if column in forecast_columns and column in training_columns
    ]
    join_columns = exact_join_columns if len(exact_join_columns) >= 3 else fallback_join_columns
    if len(join_columns) < 2:
        st.info("Holiday error comparison needs compatible horizon and timestamp columns for a clean join.")
        return None

    holiday_case = holiday_case_sql("t", training_holiday_column)
    select_columns = ", ".join(f"f.{quote_identifier(column)}" for column in group_columns)
    group_clause = ", ".join([*(f"f.{quote_identifier(column)}" for column in group_columns), holiday_case])
    join_clause = " and ".join(
        f"f.{quote_identifier(column)} = t.{quote_identifier(column)}" for column in join_columns
    )
    sql = f"""
    select
        {select_columns + "," if select_columns else ""}
        {holiday_case} as holiday_group,
        avg({error_expression}) as mae,
        count(*) as row_count,
        'ml.forecast_results + marts.mart_forecast_training_set' as source
    from ml.forecast_results f
    join marts.mart_forecast_training_set t
        on {join_clause}
    where {error_expression} is not null
    group by {group_clause}
    order by {group_clause}
    """
    return load_table(sql, "Could not calculate holiday error comparison.")


def holiday_error_section(backtests: pd.DataFrame | None) -> None:
    st.subheader("Holiday Error Comparison")
    summary = precomputed_holiday_backtest_summary(backtests)
    if summary.empty:
        summary = load_holiday_error_summary()

    if summary is None or summary.empty:
        st.info("Holiday vs non-holiday MAE is not available from the current governance or forecast outputs.")
        return

    selected = summary.copy()
    if "evaluation_split" in selected.columns:
        splits = sorted(selected["evaluation_split"].dropna().unique())
        split = st.selectbox(
            "Split",
            splits,
            index=splits.index("test") if "test" in splits else 0,
            key="governance_holiday_split",
        )
        selected = selected[selected["evaluation_split"] == split]
    selected = filter_by_horizon(selected, "governance_holiday_horizon")

    if selected.empty:
        st.info("No holiday error rows match the selected filters.")
        return

    metric_row(
        {
            "Holiday rows": f"{int(selected.loc[selected['holiday_group'] == 'Holiday', 'row_count'].sum()):,}" if "row_count" in selected.columns else "n/a",
            "Non-holiday rows": f"{int(selected.loc[selected['holiday_group'] == 'Non-holiday', 'row_count'].sum()):,}" if "row_count" in selected.columns else "n/a",
            "Source": selected["source"].iloc[0] if "source" in selected.columns else "n/a",
        }
    )
    st.plotly_chart(
        px.bar(
            selected,
            x="holiday_group",
            y="mae",
            color="model_name" if "model_name" in selected.columns else "holiday_group",
            barmode="group",
            title="Holiday vs Non-holiday MAE",
            hover_data=available_columns(selected, ["horizon_hours", "evaluation_split", "row_count", "source"]),
        ),
        width="stretch",
    )
    st.dataframe(selected, width="stretch", hide_index=True)


def model_governance() -> None:
    registry = governance_registry()
    backtests = governance_backtests()
    decisions = governance_decisions()
    importance = governance_feature_importance()

    current_champion_section(registry)
    backtest_window_section(backtests, registry)
    promotion_decision_section(decisions)
    feature_importance_section(importance)
    holiday_error_section(backtests)


PAGES = {
    "Market Overview": market_overview,
    "Load Forecast": load_forecast,
    "Model Comparison": model_comparison,
    "Model Governance": model_governance,
    "Error Analysis": error_analysis,
    "Peak Demand Monitor": peak_demand_monitor,
    "Data Quality": data_quality,
}


def main() -> None:
    st.title("AESO Load Forecasting & Market Analytics")
    page = st.sidebar.radio("Page", list(PAGES))
    PAGES[page]()


if __name__ == "__main__":
    main()
