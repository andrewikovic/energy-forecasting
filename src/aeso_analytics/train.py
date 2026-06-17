from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from xgboost import XGBRegressor

from aeso_analytics.config import (
    PROJECT_ROOT,
    get_backtest_config,
    get_promotion_rule_config,
)
from aeso_analytics.database import ensure_schemas, get_engine, replace_dataframe, write_dataframe
from aeso_analytics.features import FEATURE_COLUMNS
from aeso_analytics.governance import (
    BacktestWindow,
    apply_promotion_decisions,
    current_champion_rows,
    decide_promotions,
    generate_backtest_windows,
    normalize_model_registry,
)
from aeso_analytics.metrics import calculate_metrics


BASELINE_SPECS = {
    "baseline_same_hour_yesterday": "load_lag_24h",
    "baseline_same_hour_last_week": "load_lag_168h",
    "baseline_rolling_24h_average": "load_rolling_24h_avg",
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 4,
}


def load_training_set() -> pd.DataFrame:
    engine = get_engine()
    query = text("select * from marts.mart_forecast_training_set")
    try:
        df = pd.read_sql_query(query, engine)
    except ProgrammingError as exc:
        raise SystemExit(
            "Training mart not found. Run `make ingest` and `make dbt-run` before `make train`."
        ) from exc

    if df.empty:
        raise SystemExit("Training mart is empty. Check raw ingestion and dbt model filters.")

    for col in ("feature_as_of_timestamp_utc", "target_timestamp_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["horizon_hours", "target_timestamp_utc"]).reset_index(drop=True)


def assign_time_splits(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("target_timestamp_utc").copy()
    unique_ts = result["target_timestamp_utc"].drop_duplicates().sort_values().to_numpy()
    if len(unique_ts) < 30:
        raise ValueError("At least 30 hourly rows are required for time-based splits")

    train_end = unique_ts[int(len(unique_ts) * 0.70) - 1]
    val_end = unique_ts[int(len(unique_ts) * 0.85) - 1]
    result["evaluation_split"] = np.select(
        [
            result["target_timestamp_utc"] <= train_end,
            result["target_timestamp_utc"] <= val_end,
        ],
        ["train", "validation"],
        default="test",
    )
    return result


def load_existing_model_registry(engine) -> pd.DataFrame:
    try:
        return pd.read_sql_query(text("select * from ml.model_registry"), engine)
    except ProgrammingError:
        return pd.DataFrame()


def _json_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError, ValueError):
        return default
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return default


def build_xgboost(params: dict | None = None) -> XGBRegressor:
    model_params = dict(XGBOOST_PARAMS)
    if params:
        model_params.update(params)
    return XGBRegressor(**model_params)


def append_forecasts_and_metrics(
    model_registry_rows: list[dict],
    forecast_rows: list[pd.DataFrame],
    performance_rows: list[dict],
    model_id: str,
    model_name: str,
    horizon_hours: int,
    split_df: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    peak_threshold: float,
) -> None:
    pred = pd.Series(predictions, index=split_df.index, dtype=float)
    effective_peak_threshold = peak_threshold
    if not (split_df["target_load_mw"] >= effective_peak_threshold).any():
        effective_peak_threshold = float(split_df["target_load_mw"].quantile(0.90))

    result = split_df[
        [
            "horizon_hours",
            "evaluation_split",
            "feature_as_of_timestamp_utc",
            "target_timestamp_utc",
            "target_load_mw",
            "actual_pool_price_cad_mwh",
        ]
    ].copy()
    result["model_id"] = model_id
    result["model_name"] = model_name
    result["predicted_load_mw"] = pred
    result["error_mw"] = result["predicted_load_mw"] - result["target_load_mw"]
    result["absolute_error_mw"] = result["error_mw"].abs()
    result["absolute_percentage_error"] = (
        result["absolute_error_mw"] / result["target_load_mw"].replace(0, np.nan) * 100
    )
    result["is_peak_period"] = result["target_load_mw"] >= effective_peak_threshold
    forecast_rows.append(result)

    metrics = calculate_metrics(
        result["target_load_mw"],
        result["predicted_load_mw"],
        effective_peak_threshold,
    )
    performance_rows.append(
        {
            "model_id": model_id,
            "model_name": model_name,
            "horizon_hours": horizon_hours,
            "evaluation_split": split_df["evaluation_split"].iloc[0],
            "evaluated_at": datetime.now(timezone.utc),
            **metrics,
        }
    )


def _peak_threshold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    threshold = float(train_df["target_load_mw"].quantile(0.90))
    if not (test_df["target_load_mw"] >= threshold).any():
        threshold = float(test_df["target_load_mw"].quantile(0.90))
    return threshold


def _evaluate_backtest_window(
    model_row: pd.Series,
    horizon_df: pd.DataFrame,
    window: BacktestWindow,
    run_id: str,
    evaluated_at: datetime,
) -> dict | None:
    train_df = horizon_df[
        (horizon_df["target_timestamp_utc"] >= window.train_start)
        & (horizon_df["target_timestamp_utc"] <= window.train_end)
    ].copy()
    test_df = horizon_df[
        (horizon_df["target_timestamp_utc"] >= window.test_start)
        & (horizon_df["target_timestamp_utc"] <= window.test_end)
    ].copy()
    if train_df.empty or test_df.empty:
        return None

    model_type = str(model_row["model_type"])
    model_name = str(model_row["model_name"])
    parameters = _json_dict(model_row.get("parameters"))
    usable_test = test_df

    if model_type == "baseline":
        predictor_col = parameters.get("rule") or BASELINE_SPECS.get(model_name)
        if predictor_col is None or predictor_col not in test_df.columns:
            return None
        usable_test = test_df[test_df[predictor_col].notna()].copy()
        if usable_test.empty:
            return None
        predictions = usable_test[predictor_col]
    elif model_type == "xgboost":
        feature_names = _json_list(model_row.get("feature_names"), FEATURE_COLUMNS)
        missing_features = [column for column in feature_names if column not in horizon_df.columns]
        if missing_features:
            return None
        model = build_xgboost(parameters)
        model.fit(
            train_df[feature_names].astype(float),
            train_df["target_load_mw"].astype(float),
            verbose=False,
        )
        predictions = model.predict(usable_test[feature_names].astype(float))
    else:
        return None

    threshold = _peak_threshold(train_df, usable_test)
    metrics = calculate_metrics(
        usable_test["target_load_mw"],
        predictions,
        threshold,
    )
    return {
        "run_id": run_id,
        "model_id": model_row["model_id"],
        "model_name": model_name,
        "model_type": model_type,
        "horizon_hours": int(model_row["horizon_hours"]),
        "window_id": window.window_id,
        "backtest_mode": window.mode,
        "train_start_timestamp_utc": train_df["target_timestamp_utc"].min(),
        "train_end_timestamp_utc": train_df["target_timestamp_utc"].max(),
        "test_start_timestamp_utc": test_df["target_timestamp_utc"].min(),
        "test_end_timestamp_utc": test_df["target_timestamp_utc"].max(),
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "prediction_row_count": int(len(usable_test)),
        "peak_threshold_mw": threshold,
        "evaluated_at": evaluated_at,
        **metrics,
    }


def evaluate_backtests(
    df: pd.DataFrame,
    model_rows: pd.DataFrame,
    run_id: str,
    evaluated_at: datetime,
) -> pd.DataFrame:
    backtest_config = get_backtest_config()
    if not backtest_config.enabled or model_rows.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for horizon_hours in sorted(df["horizon_hours"].unique()):
        horizon_df = df[df["horizon_hours"] == horizon_hours].copy()
        windows = generate_backtest_windows(horizon_df, backtest_config)
        if not windows:
            continue

        horizon_model_rows = model_rows[
            model_rows["horizon_hours"].astype(int) == int(horizon_hours)
        ]
        for _, model_row in horizon_model_rows.iterrows():
            for window in windows:
                row = _evaluate_backtest_window(
                    model_row,
                    horizon_df,
                    window,
                    run_id,
                    evaluated_at,
                )
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows)


def train_models() -> None:
    df = load_training_set()
    engine = get_engine()
    ensure_schemas(engine)
    existing_registry = normalize_model_registry(load_existing_model_registry(engine))
    artifact_dir = PROJECT_ROOT / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_registry_rows: list[dict] = []
    forecast_rows: list[pd.DataFrame] = []
    performance_rows: list[dict] = []
    importance_rows: list[dict] = []

    run_ts = datetime.now(timezone.utc)
    run_label = run_ts.strftime("%Y%m%dT%H%M%SZ")
    run_id = str(uuid.uuid4())

    for horizon_hours in sorted(df["horizon_hours"].unique()):
        horizon_df = assign_time_splits(df[df["horizon_hours"] == horizon_hours].copy())
        peak_threshold = float(horizon_df["target_load_mw"].quantile(0.90))
        train_df = horizon_df[horizon_df["evaluation_split"] == "train"].copy()

        for model_name, predictor_col in BASELINE_SPECS.items():
            model_id = str(uuid.uuid4())
            model_registry_rows.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "horizon_hours": int(horizon_hours),
                    "model_type": "baseline",
                    "artifact_path": None,
                    "feature_names": json.dumps([predictor_col]),
                    "parameters": json.dumps({"rule": predictor_col}),
                    "trained_at": run_ts,
                    "train_start_timestamp_utc": train_df["target_timestamp_utc"].min(),
                    "train_end_timestamp_utc": train_df["target_timestamp_utc"].max(),
                }
            )
            for split in ("validation", "test"):
                split_df = horizon_df[
                    (horizon_df["evaluation_split"] == split) & horizon_df[predictor_col].notna()
                ].copy()
                if split_df.empty:
                    continue
                append_forecasts_and_metrics(
                    model_registry_rows,
                    forecast_rows,
                    performance_rows,
                    model_id,
                    model_name,
                    int(horizon_hours),
                    split_df,
                    split_df[predictor_col],
                    peak_threshold,
                )

        model_id = str(uuid.uuid4())
        model_name = "xgboost_regressor"
        x_train = train_df[FEATURE_COLUMNS].astype(float)
        y_train = train_df["target_load_mw"].astype(float)
        xgb = build_xgboost()
        xgb.fit(x_train, y_train, verbose=False)

        artifact_path = artifact_dir / f"xgboost_load_h{int(horizon_hours)}_{run_label}.joblib"
        joblib.dump({"model": xgb, "features": FEATURE_COLUMNS, "horizon_hours": int(horizon_hours)}, artifact_path)

        model_registry_rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "horizon_hours": int(horizon_hours),
                "model_type": "xgboost",
                "artifact_path": str(artifact_path),
                "feature_names": json.dumps(FEATURE_COLUMNS),
                "parameters": json.dumps(xgb.get_params()),
                "trained_at": run_ts,
                "train_start_timestamp_utc": train_df["target_timestamp_utc"].min(),
                "train_end_timestamp_utc": train_df["target_timestamp_utc"].max(),
            }
        )

        for feature_name, importance in zip(FEATURE_COLUMNS, xgb.feature_importances_):
            importance_rows.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "horizon_hours": int(horizon_hours),
                    "feature_name": feature_name,
                    "importance": float(importance),
                    "created_at": run_ts,
                }
            )

        for split in ("validation", "test"):
            split_df = horizon_df[horizon_df["evaluation_split"] == split].copy()
            predictions = xgb.predict(split_df[FEATURE_COLUMNS].astype(float))
            append_forecasts_and_metrics(
                model_registry_rows,
                forecast_rows,
                performance_rows,
                model_id,
                model_name,
                int(horizon_hours),
                split_df,
                predictions,
                peak_threshold,
            )

    registry_df = pd.DataFrame(model_registry_rows)
    registry_df["run_id"] = run_id
    registry_df["is_preferred"] = False
    registry_df["preferred_at"] = pd.NaT
    registry_df["retired_at"] = pd.NaT
    registry_df["promotion_decision_id"] = None
    forecasts_df = pd.concat(forecast_rows, ignore_index=True)
    performance_df = pd.DataFrame(performance_rows)
    importance_df = pd.DataFrame(importance_rows)

    champion_registry = current_champion_rows(existing_registry)
    backtest_model_rows = pd.concat(
        [champion_registry, registry_df],
        ignore_index=True,
    )
    backtest_df = evaluate_backtests(df, backtest_model_rows, run_id, run_ts)
    promotion_rules = get_promotion_rule_config()
    decisions_df, _promoted_by_horizon = decide_promotions(
        existing_registry,
        registry_df,
        backtest_df,
        promotion_rules,
        run_id,
        run_ts,
    )
    combined_registry = pd.concat(
        [existing_registry, normalize_model_registry(registry_df)],
        ignore_index=True,
    )
    combined_registry = apply_promotion_decisions(combined_registry, decisions_df, run_ts)

    replace_dataframe(engine, combined_registry, "ml", "model_registry")
    write_dataframe(engine, forecasts_df, "ml", "forecast_results", if_exists="replace")
    write_dataframe(engine, performance_df, "ml", "model_performance", if_exists="replace")
    write_dataframe(engine, importance_df, "ml", "feature_importance", if_exists="replace")
    if not backtest_df.empty:
        write_dataframe(engine, backtest_df, "ml", "model_backtest_results", if_exists="append")
    if not decisions_df.empty:
        write_dataframe(
            engine,
            decisions_df,
            "ml",
            "model_promotion_decisions",
            if_exists="append",
        )

    print(f"Registered {len(registry_df):,} new model records")
    print(f"Model registry contains {len(combined_registry):,} total records")
    print(f"Wrote {len(forecasts_df):,} forecast rows")
    print(f"Wrote {len(performance_df):,} model performance rows")
    print(f"Wrote {len(importance_df):,} feature importance rows")
    print(f"Wrote {len(backtest_df):,} model backtest rows")
    print(f"Wrote {len(decisions_df):,} promotion decision rows")


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
