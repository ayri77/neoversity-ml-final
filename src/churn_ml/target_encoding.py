from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class AutoGluonBinaryOOFTargetEncoder:
    """Compatibility extraction of the verified notebook target encoder."""

    n_splits: int = 5
    alpha: float = 10.0
    random_state: int = 42
    categorical_columns_: list[str] = field(init=False, default_factory=list)
    passthrough_columns_: list[str] = field(init=False, default_factory=list)
    encodings_: dict[str, dict[str, Any]] = field(
        init=False,
        default_factory=dict,
    )

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        self.categorical_columns_ = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        self.passthrough_columns_ = [
            column
            for column in X.columns
            if column not in self.categorical_columns_
        ]

        self.encodings_ = {}

        output = X[self.passthrough_columns_].copy()

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        splits = list(
            splitter.split(
                np.zeros(len(X)),
                y,
            )
        )

        y_array = y.reset_index(
            drop=True
        ).to_numpy(dtype=float)

        for column in self.categorical_columns_:
            values = (
                X[column]
                .reset_index(drop=True)
                .astype("object")
            )

            codes, categories = pd.factorize(
                values,
                sort=True,
            )

            valid_mask = codes >= 0
            valid_codes = codes[valid_mask]
            valid_targets = y_array[valid_mask]

            category_count = len(categories)

            counts_all = np.bincount(
                valid_codes,
                minlength=category_count,
            ).astype(float)

            sums_all = np.bincount(
                valid_codes,
                weights=valid_targets,
                minlength=category_count,
            ).astype(float)

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):
                means_all = sums_all / counts_all

            global_mean = np.nanmean(means_all)

            encoded_all = (
                means_all * counts_all
                + self.alpha * global_mean
            ) / (
                counts_all
                + self.alpha
            )

            self.encodings_[column] = {
                "categories": categories.to_numpy(copy=False),
                "encoded_values": encoded_all,
                "global_mean": float(global_mean),
            }

            oof_values = np.empty(
                len(X),
                dtype=float,
            )

            for train_index, validation_index in splits:
                train_mask = np.zeros(
                    len(X),
                    dtype=bool,
                )
                train_mask[train_index] = True

                valid_train_mask = (
                    valid_mask
                    & train_mask
                )

                train_codes = codes[valid_train_mask]
                train_targets = y_array[valid_train_mask]

                counts_train = np.bincount(
                    train_codes,
                    minlength=category_count,
                ).astype(float)

                sums_train = np.bincount(
                    train_codes,
                    weights=train_targets,
                    minlength=category_count,
                ).astype(float)

                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                ):
                    means_train = (
                        sums_train
                        / counts_train
                    )

                present_categories = counts_train > 0

                fold_global_mean = np.nanmean(
                    np.where(
                        present_categories,
                        means_train,
                        np.nan,
                    )
                )

                encoded_train = (
                    means_train * counts_train
                    + self.alpha * fold_global_mean
                ) / (
                    counts_train
                    + self.alpha
                )

                encoded_train[
                    ~present_categories
                ] = fold_global_mean

                validation_codes = codes[
                    validation_index
                ]

                fold_values = np.full(
                    len(validation_index),
                    fold_global_mean,
                    dtype=float,
                )

                known_mask = (
                    validation_codes >= 0
                )

                fold_values[known_mask] = (
                    encoded_train[
                        validation_codes[
                            known_mask
                        ]
                    ]
                )

                oof_values[
                    validation_index
                ] = fold_values

            output[
                f"{column}__te"
            ] = oof_values

        return output

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        output = X[
            self.passthrough_columns_
        ].copy()

        for column in self.categorical_columns_:
            encoding = self.encodings_[column]

            categories = encoding["categories"]
            encoded_values = encoding[
                "encoded_values"
            ]
            global_mean = encoding[
                "global_mean"
            ]

            categorical = pd.Categorical(
                X[column].astype("object"),
                categories=categories,
                ordered=False,
            )

            codes = categorical.codes

            transformed = np.full(
                len(X),
                global_mean,
                dtype=float,
            )

            known_mask = codes >= 0

            transformed[known_mask] = (
                encoded_values[
                    codes[known_mask]
                ]
            )

            output[
                f"{column}__te"
            ] = transformed

        return output
