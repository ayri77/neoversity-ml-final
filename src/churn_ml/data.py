from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

@dataclass(frozen=True)
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def load_competition_data(
    train_path: str | Path = "data/raw/final_proj_data.csv",
    test_path: str | Path = "data/raw/final_proj_test.csv",
    sample_submission_path: str | Path = (
        "data/raw/final_proj_sample_submission.csv"
    ),
    *,
    target_column: str = "y",
) -> CompetitionData:
    train_path = Path(train_path)
    test_path = Path(test_path)
    sample_submission_path = Path(sample_submission_path)

    for path in (train_path, test_path, sample_submission_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)

    if target_column not in train.columns:
        raise ValueError(
            f"Target column '{target_column}' is absent from train data."
        )

    if target_column in test.columns:
        raise ValueError(
            f"Target column '{target_column}' must not be present in test data."
        )

    feature_columns = [
        column
        for column in train.columns
        if column != target_column
    ]

    missing_in_test = sorted(set(feature_columns) - set(test.columns))
    extra_in_test = sorted(set(test.columns) - set(feature_columns))

    if missing_in_test or extra_in_test:
        raise ValueError(
            "Train and test feature schemas differ.\n"
            f"Missing in test: {missing_in_test}\n"
            f"Extra in test: {extra_in_test}"
        )

    test = test[feature_columns]

    X = train[feature_columns].copy()
    y = train[target_column].copy()

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return CompetitionData(
        train=train,
        test=test,
        sample_submission=sample_submission,
        X=X,
        y=y,
    )


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    row_count = len(df)

    records: list[dict[str, object]] = []

    for column in df.columns:
        series = df[column]

        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_rows": row_count,
                "n_missing": n_missing,
                "missing_rate": (
                    n_missing / row_count
                    if row_count > 0
                    else 0.0
                ),
                "n_unique": n_unique,
                "unique_rate": (
                    n_unique / row_count
                    if row_count > 0
                    else 0.0
                ),
                "is_constant": n_unique <= 1,
                "is_all_missing": n_missing == row_count,
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "is_integer": pd.api.types.is_integer_dtype(series),
                "is_float": pd.api.types.is_float_dtype(series),
                "is_bool": pd.api.types.is_bool_dtype(series),
                "is_object": pd.api.types.is_object_dtype(series),
                "is_category": isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                ),
            }
        )

    audit = pd.DataFrame(records)

    if audit.empty:
        return audit

    return audit.sort_values(
        by=["missing_rate", "n_unique"],
        ascending=[False, True],
    ).reset_index(drop=True)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, int | float]:
    row_count, column_count = df.shape

    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())

    return {
        "n_rows": row_count,
        "n_columns": column_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": (
            duplicate_rows / row_count
            if row_count > 0
            else 0.0
        ),
        "total_missing": total_missing,
        "columns_with_missing": int(
            df.isna().any(axis=0).sum()
        ),
        "constant_columns": int(
            (df.nunique(dropna=True) <= 1).sum()
        ),
    }

def compare_feature_distributions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    include_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare feature cardinality and value distributions in train and test."""

    def _format_feature_value(value: Any) -> str | Any:
        """Return a readable representation of a feature value."""
        return "<MISSING>" if pd.isna(value) else value  

    missing_columns = sorted(
        set(columns) - set(train.columns) - set(test.columns)
    )
    if missing_columns:
        raise KeyError(
            f"Columns not found in train or test: {missing_columns}"
        )

    summary_records: list[dict[str, object]] = []
    distribution_records: list[dict[str, object]] = []

    for column in columns:
        for dataset_name, dataframe in (
            ("train", train),
            ("test", test),
        ):
            series = dataframe[column]

            summary_records.append(
                {
                    "column": column,
                    "dataset": dataset_name,
                    "n_rows": len(series),
                    "n_missing": int(series.isna().sum()),
                    "missing_rate": float(series.isna().mean()),
                    "n_unique": int(series.nunique(dropna=True)),
                }
            )

            value_counts = series.value_counts(
                dropna=not include_missing
            )

            for value, count in value_counts.items():
                distribution_records.append(
                    {
                        "column": column,
                        "dataset": dataset_name,
                        "value": _format_feature_value(value),
                        "count": int(count),
                        "rate": float(count / len(series)),
                    }
                )

    summary = pd.DataFrame(summary_records)
    distributions = pd.DataFrame(distribution_records)

    return summary, distributions