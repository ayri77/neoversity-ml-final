"""Export the compact, tracked evidence bundle for the final project report.

The exporter reads metadata only. It never loads models or prediction arrays and
never writes inside the immutable local artifact tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "final"

DATASETS = [
    ("v0_raw_minimal", None, 205, "none", "Minimal raw baseline after removing constant and entirely missing features."),
    ("v1_missingness_summary", "v0_raw_minimal", 213, "none", "Row-level missingness summaries."),
    ("v2_missingness_indicators", "v0_raw_minimal", 393, "none", "Broad feature-level missingness indicators."),
    ("v3_targeted_missingness", "v1_missingness_summary", 217, "exploratory", "Four target-informed missingness indicators; exploratory only."),
    ("v4_zero_value_summary", "v0_raw_minimal", 209, "none", "Row-level zero-value summaries."),
    ("v5_joint_missingness_pattern", "v1_missingness_summary", 214, "none", "Joint missingness-pattern state."),
    ("v6_compact_missingness_indicators", "v1_missingness_summary", 247, "none", "Structurally unique missingness indicators."),
    ("v7_compact_zero_indicators", "v4_zero_value_summary", 234, "none", "Compact feature-level zero indicators."),
]

CANDIDATE_HASHES = {
    "pc1_d833780dcc6565cb": "597fd22cbd051badab5d017bd8f23b0019b24023e89d9cecc81e7a0d07c5f4f8",
    "pc1_c54c97e9c32f33a2": "b93b5e36a4cab6b3a951c08efc237bcce3677b221a1834e6d66897a2832c507d",
    "pc1_34ffeec20b01058f": "d5da8efb566b53fc9a2fbc8629969d467e5502e5faebd317c520a3b5cefaa05d",
    "pc1_ad1d75d8de7bd3cc": "940a3ef10e895df2db45dc3df02423cb3e305e5f4f461a675a429b4e028a7f3c",
    "pc1_82fd50055805314c": "80e87239131a13c5d3853eb410f9d767150dec326641334f502f29fb3cc0c8ab",
    "pc1_671b589ae5111b0d": "da2465c79b6706adc6251339ae0e5b1ede8ede6ac0135a6f27344aecf2b161be",
    "pc1_eef84ff4999c488a": "a213567f86da533756262d3cce5a3e45400e841973a60eb2e965f645f29bbe7e",
    "pc1_9255274a6cbd3bab": "8fd31c4ba462b234c8a2ba7d0a7a9649f343713059402a27b2b7ba884061742e",
}

SUBMISSIONS = [
    {
        "portfolio_rank": 1,
        "label": "Historical AutoGluon v3 LightGBMPrep-r31",
        "candidate_id": "pc1_d833780dcc6565cb",
        "blend_id": None,
        "dataset_id": "v3_targeted_missingness",
        "threshold": 0.117,
        "positive_count": 541,
        "internal_balanced_accuracy": 0.895158,
        "internal_protocol": "historical explicit OOF validation; descriptive, not final 5x5 meta-CV",
        "public_score": 0.9112,
        "exploratory": True,
        "submission_id": "anchor-v3-r31-reproduction-20260802",
        "submission_sha256": "8e2e91724bc5d776612d0fbd321a16a6c2d134317947197c86c30ef58cf9b4b2",
    },
    {
        "portfolio_rank": 2,
        "label": "Final A+D",
        "candidate_id": "pc1_eef84ff4999c488a",
        "blend_id": "pb1_1c6b3a8dbd482a6a",
        "dataset_id": "v3_targeted_missingness + v6_compact_missingness_indicators",
        "threshold": 0.117,
        "positive_count": 551,
        "internal_balanced_accuracy": 0.8961291445517418,
        "internal_protocol": "exploratory_confirmation_v1 honest repeated 5x5 meta-CV mean",
        "public_score": 0.9080,
        "exploratory": True,
        "submission_id": "final_confirm_a_d_20260802",
        "submission_sha256": "469827a616d15c0421aaaf7b6cd8ba2d6b6991da6646292e08c8c868d82109fe",
    },
    {
        "portfolio_rank": 3,
        "label": "Target-independent native probability blend",
        "candidate_id": "pc1_c54c97e9c32f33a2",
        "blend_id": "pb1_d047ebe82811ef1a",
        "dataset_id": "v5_joint_missingness_pattern + v6_compact_missingness_indicators",
        "threshold": 0.17,
        "positive_count": 527,
        "internal_balanced_accuracy": 0.8982157138796905,
        "internal_protocol": "native blend honest 2-repeat 5-fold meta-CV mean; separate protocol",
        "public_score": 0.9048,
        "exploratory": False,
        "submission_id": "sub_Native-blend-3-models-v5-v6_c54c97e9",
        "submission_sha256": "0a223baa665ae8fe6b9c0f4f82617e36103c50d14f6a751ce0bb6acd0e794105",
    },
    {
        "portfolio_rank": 4,
        "label": "Final A+B",
        "candidate_id": "pc1_671b589ae5111b0d",
        "blend_id": "pb1_6ec07045204974b8",
        "dataset_id": "v3_targeted_missingness + native v5/v6 blend",
        "threshold": 0.125,
        "positive_count": 567,
        "internal_balanced_accuracy": 0.8964202794136764,
        "internal_protocol": "exploratory_confirmation_v1 honest repeated 5x5 meta-CV mean",
        "public_score": 0.9042,
        "exploratory": True,
        "submission_id": "final_confirm_a_b_20260802",
        "submission_sha256": "da17ca43204eac7ecbdd9a6ef4c0814ab0d5cd575da56bb7f52d61b726441c1d",
    },
    {
        "portfolio_rank": 5,
        "label": "Final A+H",
        "candidate_id": "pc1_9255274a6cbd3bab",
        "blend_id": "pb1_1ca68e70755eea20",
        "dataset_id": "v3_targeted_missingness + v6 WeightedEnsemble",
        "threshold": 0.226,
        "positive_count": 531,
        "internal_balanced_accuracy": 0.8947804150445384,
        "internal_protocol": "exploratory_confirmation_v1 honest repeated 5x5 meta-CV mean",
        "public_score": 0.9039,
        "exploratory": True,
        "submission_id": "final_confirm_a_h_20260802",
        "submission_sha256": "58c53690d255ff76468acfd66cd796db4cd13a18de02541ffebc7181d42b10c5",
    },
]

CONFIRMATION = [
    ("baseline_a", "pb1_33938641a3a8e57a", None, 0.8928432027038044, 0.0014620403736695868, 0.8902914653464911, 0.8939704194289667, 0.895157960601834, 0.117, 541, {"pc1_d833780dcc6565cb": 1.0}),
    ("confirm_a_b", "pb1_6ec07045204974b8", "pc1_671b589ae5111b0d", 0.8964202794136764, 0.00043804765194534624, 0.8957122933645311, 0.896782843004413, 0.900249185355568, 0.125, 567, {"pc1_c54c97e9c32f33a2": 0.62, "pc1_d833780dcc6565cb": 0.38}),
    ("confirm_a_d", "pb1_1c6b3a8dbd482a6a", "pc1_eef84ff4999c488a", 0.8961291445517418, 0.001351855865539333, 0.8944659259406141, 0.897282315330738, 0.900558078254337, 0.117, 551, {"pc1_34ffeec20b01058f": 0.53, "pc1_d833780dcc6565cb": 0.47}),
    ("confirm_a_h", "pb1_1ca68e70755eea20", "pc1_9255274a6cbd3bab", 0.8947804150445384, 0.0015549776021496604, 0.8924766292337826, 0.896711898986294, 0.897593191136845, 0.226, 531, {"pc1_ad1d75d8de7bd3cc": 0.35, "pc1_d833780dcc6565cb": 0.65}),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def expected_artifacts() -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for candidate_id, expected_hash in CANDIDATE_HASHES.items():
        artifacts.append({
            "kind": "candidate_manifest",
            "id": candidate_id,
            "path": f"artifacts/prediction_candidates/{candidate_id}/candidate_manifest.json",
            "sha256": expected_hash,
        })
    artifacts.extend([
        {"kind": "campaign_plan", "id": "exploratory_confirmation_v1", "path": "artifacts/blend_campaigns/exploratory_confirmation_v1/campaign_plan.json", "identity_field": "plan_hash", "identity_value": "857e146a11c8591032fab8e8f3d1994e391fff2ef38e0cc4df2ade3a696dc172"},
        {"kind": "campaign_plan", "id": "exploratory_neural_screen_v1", "path": "artifacts/blend_campaigns/exploratory_neural_screen_v1/campaign_plan.json", "identity_field": "plan_hash", "identity_value": "05bc790326ca6419a47a46da9a016abfe17d779a66bffeca6462036186b94566"},
    ])
    for item in SUBMISSIONS:
        artifacts.append({
            "kind": "submission_csv",
            "id": item["submission_id"],
            "path": f"artifacts/candidate_submissions/{item['submission_id']}/submission.csv",
            "sha256": item["submission_sha256"],
        })
    return artifacts


def verify_local_artifacts(items: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for item in items:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
            item["verification_status"] = "missing"
            continue
        if "sha256" in item:
            actual = sha256(path)
            item["actual_sha256"] = actual
            if actual != item["sha256"]:
                errors.append(f"SHA-256 drift: {item['path']}")
                item["verification_status"] = "drift"
            else:
                item["verification_status"] = "verified"
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            actual = payload.get(item["identity_field"])
            item["actual_identity_value"] = actual
            if actual != item["identity_value"]:
                errors.append(f"identity drift: {item['path']}")
                item["verification_status"] = "drift"
            else:
                item["verification_status"] = "verified"
    if errors:
        raise SystemExit("Final evidence verification failed:\n- " + "\n- ".join(errors))


def build_evidence() -> dict[str, Any]:
    datasets = [
        {
            "dataset_id": dataset_id,
            "parent_dataset_id": parent,
            "feature_count": features,
            "train_rows": 10000,
            "test_rows": 2500,
            "target_dependency": dependency,
            "summary": summary,
            "provenance_path": f"data/processed/{dataset_id}/dataset_manifest.json",
        }
        for dataset_id, parent, features, dependency, summary in DATASETS
    ]
    confirmation = [
        {
            "experiment_id": experiment_id,
            "blend_id": blend_id,
            "canonical_candidate_id": candidate_id,
            "honest_mean_balanced_accuracy": mean,
            "honest_balanced_accuracy_std": std,
            "min_repeat_balanced_accuracy": minimum,
            "max_repeat_balanced_accuracy": maximum,
            "descriptive_full_oof_balanced_accuracy": descriptive,
            "threshold": threshold,
            "test_positive_count": positives,
            "weights": weights,
            "protocol": "exploratory_confirmation_v1: 5 folds x 5 repeats; native deterministic pair optimization; held-out threshold decisions",
            "exploratory": True,
            "provenance_path": f"artifacts/blend_campaigns/exploratory_confirmation_v1/search_results/{experiment_id}.json",
        }
        for experiment_id, blend_id, candidate_id, mean, std, minimum, maximum, descriptive, threshold, positives, weights in CONFIRMATION
    ]
    return {
        "schema_version": "final_project_evidence_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_git_commit": git_head(),
        "competition": {
            "task": "binary telecom churn classification",
            "metric": "Balanced Accuracy",
            "train_rows": 10000,
            "test_rows": 2500,
            "raw_input_features": 230,
            "positive_class_count": 1305,
            "positive_class_rate": 0.1305,
        },
        "datasets": datasets,
        "historical_anchor": {
            "candidate_id": "pc1_d833780dcc6565cb",
            "model": "LightGBMPrep_r31_BAG_L1",
            "dataset_id": "v3_targeted_missingness",
            "threshold": 0.117,
            "test_positive_count": 541,
            "validation_balanced_accuracy": 0.895158,
            "descriptive_full_oof_balanced_accuracy": 0.895157960601834,
            "kaggle_public_score": 0.9112,
            "exploratory": True,
            "metric_role": "Historical explicit OOF validation is internal evidence; Kaggle Public Score is external evidence only.",
            "provenance_path": "artifacts/prediction_candidates/pc1_d833780dcc6565cb/candidate_manifest.json",
        },
        "campaigns": {
            "confirmation": {
                "campaign_id": "exploratory_confirmation_v1",
                "plan_hash": "857e146a11c8591032fab8e8f3d1994e391fff2ef38e0cc4df2ade3a696dc172",
                "folds": 5,
                "repeats": 5,
                "seed": 42,
                "pairwise_grid_step": 0.01,
                "threshold_minimum": 0.05,
                "threshold_maximum": 0.30,
                "threshold_step": 0.001,
                "results": confirmation,
            },
            "neural_screen": {
                "campaign_id": "exploratory_neural_screen_v1",
                "plan_hash": "05bc790326ca6419a47a46da9a016abfe17d779a66bffeca6462036186b94566",
                "candidate_id": "pc1_82fd50055805314c",
                "baseline_honest_mean_balanced_accuracy": 0.8928432027038044,
                "blend_honest_mean_balanced_accuracy": 0.8924195655670344,
                "selected_weight": 0.13,
                "conclusion": "The neural candidate did not improve the anchor and was excluded from further blending.",
                "exploratory": True,
                "provenance_path": "artifacts/blend_campaigns/exploratory_neural_screen_v1/experiment_results.json",
            },
        },
        "native_blend": {
            "candidate_id": "pc1_c54c97e9c32f33a2",
            "blend_id": "pb1_d047ebe82811ef1a",
            "weights": {
                "pc1_34ffeec20b01058f": 0.88,
                "pc1_8da68e3d2071fb71": 0.10,
                "pc1_9cff07244534e636": 0.02,
            },
            "component_labels": {
                "pc1_34ffeec20b01058f": "LightGBMPrep_r31 v6",
                "pc1_8da68e3d2071fb71": "WeightedEnsemble_L2 v5",
                "pc1_9cff07244534e636": "NeuralNetTorch_r37 v6",
            },
            "threshold": 0.17,
            "test_positive_count": 527,
            "honest_mean_balanced_accuracy": 0.8982157138796905,
            "protocol": "native blend honest 2-repeat 5-fold meta-CV; not directly comparable to the final 5x5 campaign",
            "kaggle_public_score": 0.9048,
            "exploratory": False,
            "provenance_path": "artifacts/prediction_blends/pb1_d047ebe82811ef1a/evaluation.json",
        },
        "selected_submissions": SUBMISSIONS,
        "candidate_manifest_sha256": CANDIDATE_HASHES,
        "interpretation": {
            "primary": "A+B was the strongest and most stable candidate under the project-owned final 5x5 confirmation protocol.",
            "external": "A+D was the best new Public result (0.9080), while the historical AutoGluon-screened anchor remained best overall on the Public leaderboard (0.9112).",
            "caution": "Public ordering did not exactly follow honest internal evaluation. No Private Score or final rank is known.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-local-verification", action="store_true", help="Export declared tracked facts without checking ignored local artifacts.")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = expected_artifacts()
    if args.skip_local_verification:
        for item in artifacts:
            item["verification_status"] = "skipped_by_request"
    else:
        verify_local_artifacts(artifacts)
    evidence = build_evidence()
    (OUTPUT_DIR / "final_results.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        OUTPUT_DIR / "final_results.csv",
        evidence["campaigns"]["confirmation"]["results"],
        ["experiment_id", "blend_id", "canonical_candidate_id", "honest_mean_balanced_accuracy", "honest_balanced_accuracy_std", "min_repeat_balanced_accuracy", "max_repeat_balanced_accuracy", "descriptive_full_oof_balanced_accuracy", "threshold", "test_positive_count", "protocol", "exploratory", "provenance_path"],
    )
    write_csv(
        OUTPUT_DIR / "selected_submissions.csv",
        evidence["selected_submissions"],
        ["portfolio_rank", "label", "candidate_id", "blend_id", "dataset_id", "threshold", "positive_count", "internal_balanced_accuracy", "internal_protocol", "public_score", "exploratory", "submission_id", "submission_sha256"],
    )
    manifest = {
        "schema_version": "final_artifact_manifest_v1",
        "generated_at_utc": evidence["generated_at_utc"],
        "source_git_commit": evidence["source_git_commit"],
        "verification_mode": "tracked_declarations_only" if args.skip_local_verification else "local_immutable_artifacts",
        "artifacts": artifacts,
    }
    (OUTPUT_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote compact evidence to {OUTPUT_DIR.relative_to(ROOT).as_posix()}")
    print(f"Verified artifacts: {sum(item['verification_status'] == 'verified' for item in artifacts)} / {len(artifacts)}")


if __name__ == "__main__":
    main()
