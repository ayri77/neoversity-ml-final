"""Author-supplied build specification for native dataset materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.churn_ml.dataset_registry.constants import TARGET_DEPENDENCY_VALUES
from src.churn_ml.dataset_registry.errors import DatasetRegistryError
from src.churn_ml.dataset_registry.schema import TargetDependency

AUTHOR_BUILD_SPEC_KEYS = frozenset(
    {
        "dataset_id",
        "parent_dataset_id",
        "hypothesis",
        "transformations",
        "target_dependency",
        "summary_features",
        "binary_indicator_features",
        "use_missing_suffix",
    }
)

FORBIDDEN_CALCULATED_KEYS = frozenset(
    {
        "schema_version",
        "files",
        "train_row_count",
        "test_row_count",
        "n_features",
        "features",
        "schema_hash",
        "content_hashes",
        "target",
        "row_identity",
        "row_alignment_check",
        "train_content_hash",
        "test_content_hash",
        "target_hash",
        "n_train_rows",
        "n_test_rows",
        "train_path",
        "test_path",
        "target_path",
        "created_from_git_commit",
    }
)


@dataclass(frozen=True)
class DatasetBuildSpec:
    """Author-supplied fields only. Calculated manifest fields are Registry-owned."""

    dataset_id: str
    parent_dataset_id: str | None
    hypothesis: str
    transformations: tuple[Mapping[str, Any], ...]
    target_dependency: TargetDependency
    summary_features: tuple[str, ...] = ()
    binary_indicator_features: tuple[str, ...] = ()
    use_missing_suffix: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "parent_dataset_id": self.parent_dataset_id,
            "hypothesis": self.hypothesis,
            "transformations": [dict(item) for item in self.transformations],
            "target_dependency": self.target_dependency,
            "summary_features": list(self.summary_features),
            "binary_indicator_features": list(self.binary_indicator_features),
            "use_missing_suffix": self.use_missing_suffix,
        }


def parse_build_spec(value: DatasetBuildSpec | Mapping[str, Any]) -> DatasetBuildSpec:
    if isinstance(value, DatasetBuildSpec):
        return value
    if not isinstance(value, Mapping):
        raise DatasetRegistryError("build_spec must be a DatasetBuildSpec or mapping.")
    unknown = set(value) - AUTHOR_BUILD_SPEC_KEYS
    calculated = sorted(unknown & FORBIDDEN_CALCULATED_KEYS)
    if calculated:
        raise DatasetRegistryError(
            "build_spec must not supply calculated fields: "
            f"{calculated}."
        )
    if unknown:
        raise DatasetRegistryError(
            f"build_spec has unknown keys: {sorted(unknown)}."
        )
    missing = {"dataset_id", "parent_dataset_id", "hypothesis", "transformations", "target_dependency"} - set(
        value
    )
    if missing:
        raise DatasetRegistryError(
            f"build_spec is missing required keys: {sorted(missing)}."
        )
    dataset_id = _non_empty_string(value["dataset_id"], "build_spec.dataset_id")
    parent_raw = value["parent_dataset_id"]
    if parent_raw is not None and (
        not isinstance(parent_raw, str) or not parent_raw
    ):
        raise DatasetRegistryError(
            "build_spec.parent_dataset_id must be a non-empty string or null."
        )
    hypothesis = _non_empty_string(value["hypothesis"], "build_spec.hypothesis")
    transformations = parse_transformations(value["transformations"])
    target_dependency = value["target_dependency"]
    if (
        not isinstance(target_dependency, str)
        or target_dependency not in TARGET_DEPENDENCY_VALUES
    ):
        raise DatasetRegistryError(
            "build_spec.target_dependency must be one of "
            f"{sorted(TARGET_DEPENDENCY_VALUES)}."
        )
    summary_features = _string_tuple(
        value.get("summary_features", ()),
        "build_spec.summary_features",
    )
    binary_indicator_features = _string_tuple(
        value.get("binary_indicator_features", ()),
        "build_spec.binary_indicator_features",
    )
    use_missing_suffix = value.get("use_missing_suffix", False)
    if type(use_missing_suffix) is not bool:
        raise DatasetRegistryError(
            "build_spec.use_missing_suffix must be a boolean."
        )
    return DatasetBuildSpec(
        dataset_id=dataset_id,
        parent_dataset_id=parent_raw,
        hypothesis=hypothesis,
        transformations=transformations,
        target_dependency=target_dependency,  # type: ignore[arg-type]
        summary_features=summary_features,
        binary_indicator_features=binary_indicator_features,
        use_missing_suffix=use_missing_suffix,
    )


def parse_transformations(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise DatasetRegistryError(
            "transformations must be a list of mappings with a non-empty type."
        )
    parsed: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            raise DatasetRegistryError(
                f"transformations[{index}] must be a mapping with a non-empty type; "
                "free-form strings are not allowed."
            )
        if not isinstance(item, dict):
            raise DatasetRegistryError(
                f"transformations[{index}] must be a mapping with a non-empty type."
            )
        transform_type = item.get("type")
        if not isinstance(transform_type, str) or not transform_type.strip():
            raise DatasetRegistryError(
                f"transformations[{index}].type must be a non-empty string."
            )
        parsed.append(dict(item))
    return tuple(parsed)


def native_build_spec(dataset_id: str) -> DatasetBuildSpec:
    """Return the canonical author build specification for a native dataset id."""
    from src.churn_ml.dataset_registry.constants import NATIVE_CATALOG

    if dataset_id not in NATIVE_CATALOG:
        raise DatasetRegistryError(
            f"No native build specification for dataset_id {dataset_id!r}."
        )
    catalog = NATIVE_CATALOG[dataset_id]
    return DatasetBuildSpec(
        dataset_id=dataset_id,
        parent_dataset_id=catalog["parent_dataset_id"],
        hypothesis=str(catalog["hypothesis"]),
        transformations=tuple(dict(item) for item in catalog["transformations"]),
        target_dependency=catalog["target_dependency"],  # type: ignore[arg-type]
        summary_features=tuple(catalog.get("summary_features", ())),
        binary_indicator_features=tuple(
            catalog.get("binary_indicator_features", ())
        ),
        use_missing_suffix=bool(catalog.get("use_missing_suffix", False)),
    )


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetRegistryError(f"{label} must be a non-empty string.")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, tuple):
        sequence: Sequence[Any] = value
    elif isinstance(value, list):
        sequence = value
    else:
        raise DatasetRegistryError(f"{label} must be a list of strings.")
    if not all(isinstance(item, str) and item for item in sequence):
        raise DatasetRegistryError(f"{label} must be a list of non-empty strings.")
    return tuple(sequence)
