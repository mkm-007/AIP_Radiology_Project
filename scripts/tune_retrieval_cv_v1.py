# scripts/tune_retrieval_cv_v1.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"
CV_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_with_cv_folds_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH = OUTPUT_DIR / "retrieval_cv_fold_results_v1.csv"
SUMMARY_RESULTS_PATH = OUTPUT_DIR / "retrieval_cv_summary_v1.csv"
BEST_CONFIG_PATH = OUTPUT_DIR / "retrieval_cv_best_config_v1.csv"

PCA_OPTIONS = [None, 256, 128, 64]
METRIC_OPTIONS = ["cosine", "euclidean"]


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


def apply_pca(train_x, valid_x, n_components):
    if n_components is None:
        return train_x, valid_x

    max_valid_components = min(train_x.shape[0], train_x.shape[1])
    n_components = min(n_components, max_valid_components)

    pca = PCA(n_components=n_components, random_state=42)
    train_out = pca.fit_transform(train_x)
    valid_out = pca.transform(valid_x)
    return train_out, valid_out


def retrieve_predictions(train_meta, valid_meta, train_feats, valid_feats, metric):
    rows = []

    for region in sorted(valid_meta["canonical_region"].unique()):
        valid_region_idx = np.where(valid_meta["canonical_region"].values == region)[0]
        train_region_idx = np.where(train_meta["canonical_region"].values == region)[0]

        if len(valid_region_idx) == 0 or len(train_region_idx) == 0:
            continue

        vr = valid_feats[valid_region_idx]
        tr = train_feats[train_region_idx]

        if metric == "cosine":
            sim = cosine_similarity(vr, tr)
            best_idx = sim.argmax(axis=1)
            best_score = sim.max(axis=1)
        elif metric == "euclidean":
            dist = euclidean_distances(vr, tr)
            best_idx = dist.argmin(axis=1)
            best_score = -dist.min(axis=1)   # negative distance so larger is "better"
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        for i, valid_local_idx in enumerate(valid_region_idx):
            matched_train_idx = train_region_idx[best_idx[i]]

            valid_row = valid_meta.iloc[valid_local_idx]
            train_row = train_meta.iloc[matched_train_idx]

            gt_text = str(valid_row["region_text_target"])
            pred_text = str(train_row["region_text_target"])

            rows.append({
                "canonical_region": region,
                "ground_truth_text": gt_text,
                "predicted_text": pred_text,
                "score": float(best_score[i]),
                "exact_match": exact_match(gt_text, pred_text),
                "jaccard_similarity": jaccard_similarity(gt_text, pred_text),
            })

    return pd.DataFrame(rows)


def main():
    print("=== TUNE RETRIEVAL CV V1 ===")

    features = np.load(FEATURES_PATH)
    feature_meta = pd.read_csv(METADATA_PATH)
    cv_df = pd.read_csv(CV_DATASET_PATH)

    # Merge cv_fold onto feature metadata using identifiers
    merge_cols = ["subject_id", "study_id", "dicom_id", "official_split", "canonical_region", "region_text_target"]
    merged = feature_meta.merge(
        cv_df[merge_cols + ["cv_fold"]],
        on=merge_cols,
        how="inner"
    )

    print(f"Feature rows loaded : {len(feature_meta)}")
    print(f"CV rows merged      : {len(merged)}")
    print(f"Feature shape       : {features.shape}")

    if len(merged) != len(features):
        raise ValueError("Merged metadata row count does not match feature row count.")

    # Use only training rows that have CV folds
    train_mask = merged["official_split"] == "train"
    train_df = merged[train_mask].copy().reset_index(drop=True)
    train_feats = features[train_mask.values]

    print(f"Training rows used for CV tuning: {len(train_df)}")

    fold_results = []

    for pca_dim in PCA_OPTIONS:
        for metric in METRIC_OPTIONS:
            print(f"\nRunning config: pca_dim={pca_dim}, metric={metric}")

            for fold in sorted(train_df["cv_fold"].dropna().unique()):
                fold = int(fold)

                fold_train_mask = train_df["cv_fold"] != fold
                fold_valid_mask = train_df["cv_fold"] == fold

                fold_train_meta = train_df[fold_train_mask].reset_index(drop=True)
                fold_valid_meta = train_df[fold_valid_mask].reset_index(drop=True)

                fold_train_feats = train_feats[fold_train_mask.values]
                fold_valid_feats = train_feats[fold_valid_mask.values]

                # Normalize before PCA/metric
                fold_train_feats = normalize_rows(fold_train_feats)
                fold_valid_feats = normalize_rows(fold_valid_feats)

                fold_train_feats, fold_valid_feats = apply_pca(
                    fold_train_feats, fold_valid_feats, pca_dim
                )

                if metric == "cosine":
                    fold_train_feats = normalize_rows(fold_train_feats)
                    fold_valid_feats = normalize_rows(fold_valid_feats)

                pred_df = retrieve_predictions(
                    fold_train_meta, fold_valid_meta,
                    fold_train_feats, fold_valid_feats,
                    metric=metric
                )

                mean_jaccard = pred_df["jaccard_similarity"].mean()
                exact_match_rate = pred_df["exact_match"].mean()
                num_eval_rows = len(pred_df)

                fold_results.append({
                    "pca_dim": "none" if pca_dim is None else pca_dim,
                    "metric": metric,
                    "cv_fold": fold,
                    "num_eval_rows": num_eval_rows,
                    "mean_jaccard_similarity": mean_jaccard,
                    "exact_match_rate": exact_match_rate
                })

                print(
                    f"  Fold {fold}: "
                    f"rows={num_eval_rows}, "
                    f"mean_jaccard={mean_jaccard:.4f}, "
                    f"exact_match={exact_match_rate:.4f}"
                )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(FOLD_RESULTS_PATH, index=False)

    summary_df = (
        fold_df.groupby(["pca_dim", "metric"])
        .agg(
            mean_jaccard_mean=("mean_jaccard_similarity", "mean"),
            mean_jaccard_std=("mean_jaccard_similarity", "std"),
            exact_match_mean=("exact_match_rate", "mean"),
            exact_match_std=("exact_match_rate", "std"),
            total_eval_rows=("num_eval_rows", "sum")
        )
        .reset_index()
        .sort_values("mean_jaccard_mean", ascending=False)
    )
    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    best_row = summary_df.iloc[[0]].copy()
    best_row.to_csv(BEST_CONFIG_PATH, index=False)

    print("\n=== CV SUMMARY ===")
    print(summary_df)

    print("\n=== BEST CONFIG ===")
    print(best_row)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Fold results : {FOLD_RESULTS_PATH}")
    print(f"Summary table: {SUMMARY_RESULTS_PATH}")
    print(f"Best config  : {BEST_CONFIG_PATH}")
    print("Retrieval CV tuning completed successfully.")


if __name__ == "__main__":
    main()