# scripts/build_cv_folds_v1.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET_PATH = OUTPUT_DIR / "modeling_dataset_with_cv_folds_v1.csv"
SUMMARY_PATH = TABLES_DIR / "cv_fold_summary_v1.csv"

N_SPLITS = 5


def main():
    print("=== BUILD CV FOLDS V1 ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded modeling dataset rows: {len(df)}")

    # Only training rows get CV folds
    train_df = df[df["official_split"] == "train"].copy().reset_index(drop=True)
    non_train_df = df[df["official_split"] != "train"].copy()

    print(f"Training rows for CV : {len(train_df)}")
    print(f"Non-train rows kept  : {len(non_train_df)}")

    if "study_id" not in train_df.columns:
        raise ValueError("Column 'study_id' is required for grouped CV.")

    gkf = GroupKFold(n_splits=N_SPLITS)

    train_df["cv_fold"] = -1

    groups = train_df["study_id"].values

    for fold_idx, (_, valid_idx) in enumerate(gkf.split(train_df, groups=groups)):
        train_df.loc[valid_idx, "cv_fold"] = fold_idx

    if (train_df["cv_fold"] == -1).any():
        raise ValueError("Some training rows were not assigned a CV fold.")

    # Non-train rows keep fold as NA
    non_train_df["cv_fold"] = pd.NA

    full_df = pd.concat([train_df, non_train_df], ignore_index=True)
    full_df.to_csv(OUTPUT_DATASET_PATH, index=False)

    # Summary table
    summary_df = (
        train_df.groupby(["cv_fold", "canonical_region"])
        .size()
        .reset_index(name="row_count")
        .sort_values(["cv_fold", "canonical_region"])
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nTraining fold counts by region:")
    print(summary_df)

    print("\nOverall fold sizes:")
    print(train_df["cv_fold"].value_counts().sort_index())

    print("\n=== SAVED OUTPUTS ===")
    print(f"Dataset with folds : {OUTPUT_DATASET_PATH}")
    print(f"Summary table      : {SUMMARY_PATH}")
    print("CV fold assignment completed successfully.")


if __name__ == "__main__":
    main()