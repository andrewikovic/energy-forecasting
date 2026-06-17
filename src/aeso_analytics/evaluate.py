from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from aeso_analytics.database import get_engine


def main() -> None:
    engine = get_engine()
    try:
        performance = pd.read_sql_query(
            text(
                """
                select
                    horizon_hours,
                    evaluation_split,
                    model_name,
                    mae,
                    rmse,
                    mape,
                    mean_error,
                    median_absolute_error,
                    underprediction_rate,
                    peak_period_mae
                from ml.model_performance
                order by horizon_hours, evaluation_split, mae
                """
            ),
            engine,
        )
    except ProgrammingError as exc:
        raise SystemExit("No model performance table found. Run `make train` first.") from exc

    if performance.empty:
        raise SystemExit("ml.model_performance is empty. Run `make train` first.")

    print(performance.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    best = (
        performance[performance["evaluation_split"] == "test"]
        .sort_values(["horizon_hours", "mae"])
        .groupby("horizon_hours")
        .head(1)
    )
    print("\nBest test models by MAE:")
    print(best[["horizon_hours", "model_name", "mae", "rmse", "mape"]].to_string(index=False))


if __name__ == "__main__":
    main()
