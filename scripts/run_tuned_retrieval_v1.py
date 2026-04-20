# scripts/run_tuned_retrieval_v1.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = OUTPUT_DIR / "tuned_retrieval_predictions_v1.csv"
SUMMARY_CSV = OUTPUT_DIR / "tuned_retrieval_summary_v1.csv"

BEST_PCA_DIM = 128
BEST_METRIC = "euclidean"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def token_set(text: str):
    return set(str(text).lower().strip().split())


def jaccard_similarity(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    if len(sa) == 0 or len(sb) == 0:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def exact_match(a: str, b: str) -> int:
    return int(str(a).strip().lower() == str(b).strip().lower())


def apply_pca(train_x, eval_x, n_components):
    max_valid_components = min(train_x.shape[0], train_x.shape[1])
    n_components = min(n_components, max_valid_components)

    pca = PCA(n_components=n_components, random_state=42)
    train_out = pca.fit_transform(train_x)
    eval_out = pca.transform(eval_x)
    return train_out, eval_out


def main():
    print("=== RUN TUNED RETRIEVAL V1 ===")

    features = np.load(FEATURES_PATH)
    meta = pd.read_csv(METADATA_PATH)

    print(f"Loaded feature array shape: {features.shape}")
    print(f"Loaded metadata rows      : {len(meta)}")

    if len(meta) != len(features):
        raise ValueError("Feature rows and metadata rows do not match.")

    train_mask = meta["official_split"] == "train"
    eval_mask = meta["official_split"].isin(["valid", "test"])

    train_meta = meta[train_mask].copy().reset_index(drop=True)
    eval_meta = meta[eval_mask].copy().reset_index(drop=True)

    train_feats = features[train_mask.values]
    eval_feats = features[eval_mask.values]

    print(f"Train rows: {len(train_meta)}")
    print(f"Eval rows : {len(eval_meta)}")

    # Normalize first, then PCA, then retrieval with euclidean
    train_feats = normalize_rows(train_feats)
    eval_feats = normalize_rows(eval_feats)

    train_feats, eval_feats = apply_pca(train_feats, eval_feats, BEST_PCA_DIM)

    prediction_rows = []

    for region in sorted(eval_meta["canonical_region"].unique()):
        eval_region_idx = np.where(eval_meta["canonical_region"].values == region)[0]
        train_region_idx = np.where(train_meta["canonical_region"].values == region)[0]

        if len(eval_region_idx) == 0 or len(train_region_idx) == 0:
            continue

        region_eval_feats = eval_feats[eval_region_idx]
        region_train_feats = train_feats[train_region_idx]

        dist = euclidean_distances(region_eval_feats, region_train_feats)
        best_train_local_idx = dist.argmin(axis=1)
        best_scores = dist.min(axis=1)

        for i, eval_local_idx in enumerate(eval_region_idx):
            matched_train_idx = train_region_idx[best_train_local_idx[i]]

            eval_row = eval_meta.iloc[eval_local_idx]
            train_row = train_meta.iloc[matched_train_idx]

            gt_text = str(eval_row["region_text_target"])
            pred_text = str(train_row["region_text_target"])

            prediction_rows.append({
                "official_split": eval_row["official_split"],
                "canonical_region": eval_row["canonical_region"],
                "query_dicom_id": eval_row["dicom_id"],
                "query_crop_path": eval_row["crop_local_path"],
                "ground_truth_text": gt_text,
                "predicted_text": pred_text,
                "matched_train_dicom_id": train_row["dicom_id"],
                "matched_train_crop_path": train_row["crop_local_path"],
                "euclidean_distance": float(best_scores[i]),
                "exact_match": exact_match(gt_text, pred_text),
                "jaccard_similarity": jaccard_similarity(gt_text, pred_text),
            })

    pred_df = pd.DataFrame(prediction_rows)

    if pred_df.empty:
        raise ValueError("No tuned retrieval predictions were generated.")

    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    summary_df = (
        pred_df.groupby(["official_split", "canonical_region"])
        .agg(
            row_count=("query_dicom_id", "count"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
            exact_match_rate=("exact_match", "mean"),
            mean_jaccard_similarity=("jaccard_similarity", "mean")
        )
        .reset_index()
        .sort_values(["official_split", "canonical_region"])
    )

    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nSummary by split and region:")
    print(summary_df)

    print("\nSample predictions:")
    sample_cols = [
        "official_split", "canonical_region",
        "ground_truth_text", "predicted_text",
        "euclidean_distance", "exact_match", "jaccard_similarity"
    ]
    print(pred_df[sample_cols].head(10))

    print("\n=== SAVED OUTPUTS ===")
    print(f"Predictions CSV : {PREDICTIONS_CSV}")
    print(f"Summary CSV     : {SUMMARY_CSV}")
    print("Tuned retrieval completed successfully.")


if __name__ == "__main__":
    main()