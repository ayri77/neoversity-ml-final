from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_numeric_distribution(
    dataframe: pd.DataFrame,
    column: str,
    *,
    bins: int = 50,
    include_log: bool = False,
    figsize: tuple[float, float] = (15, 8),
) -> Figure:
    """
    Plot the distribution of a numerical feature.

    The figure always includes:
    - a histogram of the original non-missing values;
    - a horizontal boxplot of the original non-missing values.

    When include_log is True and all non-missing values are non-negative,
    the figure also includes:
    - a histogram after log1p transformation;
    - a horizontal boxplot after log1p transformation.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    column:
        Name of the numerical feature to visualize.

    bins:
        Number of histogram bins.

    include_log:
        Whether to include log1p-transformed plots when the feature
        contains only non-negative values.

    figsize:
        Figure size.

    Returns
    -------
    Figure
        The generated matplotlib figure.
    """
    if column not in dataframe.columns:
        raise KeyError(f"Column not found: {column}")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"Column must be numerical: {column}")

    values = dataframe[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column contains no non-missing values: {column}"
        )

    use_log = include_log and values.min() >= 0

    if use_log:
        transformed_values = np.log1p(values)

        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=figsize,
        )

        axes[0, 0].hist(values, bins=bins)
        axes[0, 0].set_title(f"{column}: original distribution")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")

        axes[0, 1].hist(transformed_values, bins=bins)
        axes[0, 1].set_title(f"{column}: log1p distribution")
        axes[0, 1].set_xlabel("log1p(value)")
        axes[0, 1].set_ylabel("Frequency")

        axes[1, 0].boxplot(values, vert=False)
        axes[1, 0].set_title(f"{column}: original boxplot")
        axes[1, 0].set_xlabel("Value")

        axes[1, 1].boxplot(transformed_values, vert=False)
        axes[1, 1].set_title(f"{column}: log1p boxplot")
        axes[1, 1].set_xlabel("log1p(value)")

    else:
        figure, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
        )

        axes[0].hist(values, bins=bins)
        axes[0].set_title(f"{column}: original distribution")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(values, vert=False)
        axes[1].set_title(f"{column}: original boxplot")
        axes[1].set_xlabel("Value")

    missing_count = int(dataframe[column].isna().sum())
    missing_rate = dataframe[column].isna().mean()
    unique_count = int(values.nunique())
    skewness = values.skew()

    statistics_text = (
        f"Non-missing: {len(values):,} | "
        f"Missing: {missing_count:,} ({missing_rate:.1%}) | "
        f"Unique: {unique_count:,} | "
        f"Skewness: {skewness:.2f}"
    )

    figure.suptitle(
        f"Numerical feature distribution — {column}\n"
        f"{statistics_text}"
    )
    figure.tight_layout()

    return figure

def plot_categorical_distribution(
    dataframe: pd.DataFrame,
    column: str,
    *,
    top_n: int = 15,
    normalize: bool = True,
    include_missing: bool = True,
    group_remaining: bool = True,
    figsize: tuple[float, float] = (12, 7),
) -> Figure:
    """
    Plot the frequency distribution of a categorical feature.

    The most frequent categories are displayed as a horizontal bar chart.
    Categories outside the selected top-N can be combined into an
    ``<OTHER>`` group. Missing values can be displayed as a separate
    ``<MISSING>`` category.

    Parameters
    ----------
    dataframe:
        Source dataframe.

    column:
        Name of the categorical feature to visualize.

    top_n:
        Maximum number of original categories to display.

    normalize:
        Whether to display percentages instead of absolute counts.

    include_missing:
        Whether to display missing values as a separate category.

    group_remaining:
        Whether to combine categories outside the top-N into ``<OTHER>``.

    figsize:
        Figure size.

    Returns
    -------
    Figure
        The generated matplotlib figure.
    """
    if column not in dataframe.columns:
        raise KeyError(f"Column not found: {column}")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    series = dataframe[column]

    if not (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        raise TypeError(f"Column must be categorical: {column}")

    values = series.astype("object")

    if include_missing:
        values = values.fillna("<MISSING>")
    else:
        values = values.dropna()

    if values.empty:
        raise ValueError(f"Column contains no values to plot: {column}")

    value_counts = values.value_counts(dropna=False)

    visible_counts = value_counts.head(top_n).copy()
    remaining_count = int(value_counts.iloc[top_n:].sum())

    if group_remaining and remaining_count > 0:
        visible_counts.loc["<OTHER>"] = remaining_count

    if normalize:
        plot_values = visible_counts / len(series)
        x_label = "Share of rows"
    else:
        plot_values = visible_counts
        x_label = "Count"

    # Keep the largest category at the top of the horizontal chart.
    plot_values = plot_values.sort_values(ascending=False)

    missing_count = int(series.isna().sum())
    missing_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))

    top_non_missing = series.value_counts(dropna=True)

    if top_non_missing.empty:
        top_category = None
        top_rate = 0.0
    else:
        top_category = top_non_missing.index[0]
        top_rate = float(top_non_missing.iloc[0] / len(series))

    statistics_text = (
        f"Rows: {len(series):,} | "
        f"Missing: {missing_count:,} ({missing_rate:.1%}) | "
        f"Unique: {unique_count:,} | "
        f"Top category: {top_category!r} ({top_rate:.1%})"
    )

    figure, axis = plt.subplots(figsize=figsize)

    bars = axis.barh(
        y=plot_values.index.astype(str).tolist(),
        width=plot_values.to_numpy(),
    )
    axis.invert_yaxis()

    for bar, value in zip(bars, plot_values.values):
        label = f"{value:.1%}" if normalize else f"{int(value):,}"

        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {label}",
            va="center",
            fontsize=9,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Category")

    if normalize:
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0%}")
        )

    # Leave space for labels placed after the bars.
    axis.margins(x=0.08)

    figure.suptitle(
        f"Categorical feature distribution — {column}",
        fontsize=14,
    )

    axis.set_title(
        statistics_text,
        fontsize=11,
        pad=12,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.94))

    return figure