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

def analyze_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return feature-level missing value statistics."""

    n_rows = len(dataframe)

    summary = pd.DataFrame(
        {
            "column": dataframe.columns,
            "n_rows": n_rows,
            "n_missing": dataframe.isna().sum().values,
        }
    )

    summary["n_present"] = (
        summary["n_rows"] - summary["n_missing"]
    )

    summary["missing_rate"] = (
        summary["n_missing"] / summary["n_rows"]
    )

    summary["missing_bucket"] = (
        summary["missing_rate"]
        .map(categorize_missing_rate)
    )    

    summary = summary.sort_values(
        ["missing_rate", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary

def categorize_missing_rate(
    missing_rate: float,
) -> str:
    """Assign a readable missing-value bucket."""

    if missing_rate == 0:
        return "No missing"
    if missing_rate < 0.25:
        return "Low (<25%)"
    if missing_rate < 0.50:
        return "Moderate (25–50%)"
    if missing_rate < 0.75:
        return "High (50–75%)"
    if missing_rate < 1:
        return "Very high (75–100%)"
    return "All missing"

def analyze_feature_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a summary of feature data types.

    The Numeric category includes both integer and floating-point columns.
    """
    type_summary = pd.DataFrame(
        {
            "feature_type": [
                "Numeric",
                "Integer",
                "Float",
                "Boolean",
                "Object",
                "Category",
            ],
            "feature_count": [
                dataframe.select_dtypes(include="number").shape[1],
                dataframe.select_dtypes(include="integer").shape[1],
                dataframe.select_dtypes(include="float").shape[1],
                dataframe.select_dtypes(include="bool").shape[1],
                dataframe.select_dtypes(include="object").shape[1],
                dataframe.select_dtypes(include="category").shape[1],
            ],
        }
    )

    type_summary["feature_rate"] = (
        type_summary["feature_count"] / dataframe.shape[1]
    )

    type_summary["feature_rate_pct"] = (
        type_summary["feature_rate"] * 100
    ).round(1)    

    type_summary["description"] = [
        "Integer and floating-point features",
        "Integer features",
        "Floating-point features",
        "Boolean features",
        "Object (string) features",
        "Pandas categorical features",
    ]            

    return type_summary

def categorize_cardinality(
    n_unique: int,
) -> str:
    """Assign a readable cardinality bucket."""

    if n_unique == 0:
        return "Empty"
    if n_unique == 1:
        return "Constant"
    if n_unique == 2:
        return "Binary"
    if n_unique <= 10:
        return "Very low (3–10)"
    if n_unique <= 50:
        return "Low (11–50)"
    if n_unique <= 200:
        return "Medium (51–200)"
    return "High (>200)"

def analyze_cardinality(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return feature-level cardinality statistics."""

    n_rows = len(dataframe)

    summary = pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": dataframe.dtypes.astype(str).values,
            "n_unique": dataframe.nunique(dropna=True).values,
        }
    )

    summary["unique_rate"] = (
        summary["n_unique"] / n_rows
        if n_rows > 0
        else 0.0
    )

    summary["is_constant"] = (
        summary["n_unique"] <= 1
    )

    summary["cardinality_bucket"] = (
        summary["n_unique"]
        .map(categorize_cardinality)
    )

    summary = summary.sort_values(
        ["n_unique", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary

def analyze_numeric_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return descriptive statistics for numerical features.

    Missing values are ignored by the underlying pandas statistics.
    """

    numeric_data = dataframe.select_dtypes(include="number")

    summary = pd.DataFrame(
        {
            "column": numeric_data.columns,
            "dtype": numeric_data.dtypes.astype(str).values,
            "n_rows": len(numeric_data),
            "n_missing": numeric_data.isna().sum().values,
            "n_unique": numeric_data.nunique(dropna=True).values,
            "mean": numeric_data.mean().values,
            "std": numeric_data.std().values,
            "min": numeric_data.min().values,
            "q1": numeric_data.quantile(0.25).values,
            "median": numeric_data.median().values,
            "q3": numeric_data.quantile(0.75).values,
            "max": numeric_data.max().values,
            "skewness": numeric_data.skew().values,
            "kurtosis": numeric_data.kurt().values,
        }
    )

    summary["missing_rate"] = (
        summary["n_missing"] / summary["n_rows"]
        if len(numeric_data) > 0
        else 0.0
    )

    summary["iqr"] = summary["q3"] - summary["q1"]

    summary["range"] = summary["max"] - summary["min"]

    summary["abs_skewness"] = summary["skewness"].abs()

    summary = summary.sort_values(
        ["abs_skewness", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary

def analyze_categorical_features(
    dataframe: pd.DataFrame,
    *,
    rare_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Return summary statistics for categorical features.

    Object and pandas category columns are treated as categorical.
    Missing values are reported separately and are not included when
    determining the most frequent category or rare-category counts.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    rare_threshold:
        Maximum frequency used to classify a non-missing category as rare.
        For example, 0.01 corresponds to 1% of all dataframe rows.

    Returns
    -------
    pd.DataFrame
        One row per categorical feature.
    """
    if not 0 <= rare_threshold <= 1:
        raise ValueError("rare_threshold must be between 0 and 1.")

    categorical_data = dataframe.select_dtypes(
        include=["object", "category"]
    )

    records: list[dict[str, object]] = []

    for column in categorical_data.columns:
        series = categorical_data[column]
        non_missing = series.dropna()
        value_counts = non_missing.value_counts(dropna=True)

        n_rows = len(series)
        n_missing = int(series.isna().sum())
        n_present = int(series.notna().sum())
        n_unique = int(non_missing.nunique())

        if value_counts.empty:
            top_category = None
            top_count = 0
            top_rate = 0.0
            rare_category_count = 0
            rare_value_count = 0
        else:
            top_category = value_counts.index[0]
            top_count = int(value_counts.iloc[0])
            top_rate = top_count / n_rows

            rare_mask = (value_counts / n_rows) < rare_threshold
            rare_category_count = int(rare_mask.sum())
            rare_value_count = int(value_counts[rare_mask].sum())

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_rows": n_rows,
                "n_present": n_present,
                "n_missing": n_missing,
                "missing_rate": n_missing / n_rows,
                "n_unique": n_unique,
                "top_category": top_category,
                "top_count": top_count,
                "top_rate": top_rate,
                "rare_category_count": rare_category_count,
                "rare_value_count": rare_value_count,
                "rare_value_rate": rare_value_count / n_rows,
            }
        )

    summary = pd.DataFrame.from_records(records)

    if summary.empty:
        return summary

    return (
        summary
        .sort_values(
            ["n_unique", "missing_rate", "column"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

def categorize_top_rate(rate: float) -> str:
    """
    Categorize the dominance of the most frequent category.
    """

    if rate >= 0.95:
        return "Very dominant (>95%)"
    if rate >= 0.80:
        return "Dominant (80–95%)"
    if rate >= 0.50:
        return "Moderately dominant (50–80%)"
    return "Balanced (<50%)"