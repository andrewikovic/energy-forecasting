from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from xgboost import XGBRegressor

from aeso_analytics.config import PROJECT_ROOT
from aeso_analytics.database import ensure_schemas, get_engine, write_dataframe
from aeso_analytics.features import FEATURE_COLUMNS
from aeso_analytics.metrics import calculate_metrics


BASELINE_SPECS = {
    "baseline_same_hour_yesterday": "load_lag_24h",
    "baseline_same_hour_last_week": "load_lag_168h",
    "baseline_rolling_24h_average": "load_rolling_24h_avg",
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


def train_models() -> None:
    df = load_training_set()
    engine = get_engine()
    ensure_schemas(engine)
    artifact_dir = PROJECT_ROOT / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_registry_rows: list[dict] = []
    forecast_rows: list[pd.DataFrame] = []
    performance_rows: list[dict] = []
    importance_rows: list[dict] = []

    run_ts = datetime.now(timezone.utc)
    run_label = run_ts.strftime("%Y%m%dT%H%M%SZ")

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
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
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
    forecasts_df = pd.concat(forecast_rows, ignore_index=True)
    performance_df = pd.DataFrame(performance_rows)
    importance_df = pd.DataFrame(importance_rows)

    write_dataframe(engine, registry_df, "ml", "model_registry", if_exists="replace")
    write_dataframe(engine, forecasts_df, "ml", "forecast_results", if_exists="replace")
    write_dataframe(engine, performance_df, "ml", "model_performance", if_exists="replace")
    write_dataframe(engine, importance_df, "ml", "feature_importance", if_exists="replace")

    print(f"Registered {len(registry_df):,} model records")
    print(f"Wrote {len(forecasts_df):,} forecast rows")
    print(f"Wrote {len(performance_df):,} model performance rows")
    print(f"Wrote {len(importance_df):,} feature importance rows")


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
