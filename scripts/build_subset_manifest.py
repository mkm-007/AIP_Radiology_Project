# scripts/build_subset_manifest.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_MIMIC_DIR = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg"
RAW_CHEST_DIR = PROJECT_ROOT / "data" / "raw" / "chest_imagenome"

REPORTS_DIR = PROJECT_ROOT / "data" / "interim" / "mimic_cxr_reports_extracted"
SCENE_GRAPH_DIR = PROJECT_ROOT / "data" / "interim" / "chest_imagenome_scene_graphs"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Target subset sizes for the one-week project
TARGET_COUNTS = {
    "train": 1200,
    "valid": 150,
    "test": 150,
}

RANDOM_SEED = 42


def make_report_rel_path(subject_id, study_id):
    subject_str = str(subject_id)
    outer = f"p{subject_str[:2]}"
    inner = f"p{subject_str}"
    return Path("files") / outer / inner / f"s{study_id}.txt"


def make_scene_graph_rel_path(dicom_id):
    return Path("scene_graph") / f"{dicom_id}_SceneGraph.json"


def file_exists(base_dir: Path, rel_path: Path):
    return (base_dir / rel_path).exists()


def main():
    print("=== BUILD SUBSET MANIFEST ===")

    # ----------------------------
    # 1. Load Chest ImaGenome splits
    # ----------------------------
    train_df = pd.read_csv(RAW_CHEST_DIR / "train.csv")
    valid_df = pd.read_csv(RAW_CHEST_DIR / "valid.csv")
    test_df = pd.read_csv(RAW_CHEST_DIR / "test.csv")
    avoid_df = pd.read_csv(RAW_CHEST_DIR / "images_to_avoid.csv")

    train_df["official_split"] = "train"
    valid_df["official_split"] = "valid"
    test_df["official_split"] = "test"

    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(f"Loaded Chest ImaGenome split rows: {len(full_df)}")

    # ----------------------------
    # 2. Remove images_to_avoid only from train/valid
    # ----------------------------
    avoid_ids = set(avoid_df["dicom_id"].astype(str).tolist())

    before_avoid = len(full_df)

    mask_train_valid = full_df["official_split"].isin(["train", "valid"])
    mask_avoid = full_df["dicom_id"].astype(str).isin(avoid_ids)

    full_df = full_df[~(mask_train_valid & mask_avoid)].copy()
    after_avoid = len(full_df)

    print(f"Removed train/valid images_to_avoid rows: {before_avoid - after_avoid}")

    # ----------------------------
    # 3. Keep only frontal views (AP/PA)
    # ----------------------------
    before_frontal = len(full_df)
    full_df = full_df[full_df["ViewPosition"].isin(["AP", "PA"])].copy()
    after_frontal = len(full_df)

    print(f"Removed non-frontal rows: {before_frontal - after_frontal}")
    print(f"Remaining frontal rows: {after_frontal}")

    # ----------------------------
    # 4. Build report and scene graph paths
    # ----------------------------
    full_df["report_rel_path"] = full_df.apply(
        lambda row: str(make_report_rel_path(row["subject_id"], row["study_id"])),
        axis=1
    )
    full_df["scene_graph_rel_path"] = full_df["dicom_id"].apply(
        lambda x: str(make_scene_graph_rel_path(x))
    )

    full_df["report_exists"] = full_df["report_rel_path"].apply(
        lambda x: file_exists(REPORTS_DIR, Path(x))
    )
    full_df["scene_graph_exists"] = full_df["scene_graph_rel_path"].apply(
        lambda x: file_exists(SCENE_GRAPH_DIR, Path(x))
    )

    # ----------------------------
    # 5. Merge MIMIC metadata and split info
    # ----------------------------
    metadata_df = pd.read_csv(
        RAW_MIMIC_DIR / "mimic-cxr-2.0.0-metadata.csv.gz",
        compression="gzip"
    )
    mimic_split_df = pd.read_csv(
        RAW_MIMIC_DIR / "mimic-cxr-2.0.0-split.csv.gz",
        compression="gzip"
    )

    merge_cols = ["dicom_id", "subject_id", "study_id"]
    full_df = full_df.merge(
        metadata_df[merge_cols + ["ViewPosition"]],
        on=merge_cols,
        how="left",
        suffixes=("", "_mimic_meta")
    )

    full_df = full_df.merge(
        mimic_split_df[merge_cols + ["split"]],
        on=merge_cols,
        how="left"
    )

    # ----------------------------
    # 6. Keep only rows with both report and scene graph
    # ----------------------------
    before_exists = len(full_df)
    full_df = full_df[(full_df["report_exists"]) & (full_df["scene_graph_exists"])].copy()
    after_exists = len(full_df)

    print(f"Removed rows missing report or scene graph: {before_exists - after_exists}")
    print(f"Rows remaining after existence checks: {after_exists}")

    # ----------------------------
    # 7. Downsample each split
    # ----------------------------
    subset_parts = []
    subset_summary = []

    for split_name, target_n in TARGET_COUNTS.items():
        split_rows = full_df[full_df["official_split"] == split_name].copy()
        available_n = len(split_rows)
        actual_n = min(target_n, available_n)

        split_rows = split_rows.sample(n=actual_n, random_state=RANDOM_SEED).copy()
        subset_parts.append(split_rows)

        subset_summary.append({
            "split": split_name,
            "available_after_filtering": available_n,
            "target_requested": target_n,
            "actual_selected": actual_n,
        })

        print(f"{split_name}: available={available_n}, selected={actual_n}")

    subset_df = pd.concat(subset_parts, ignore_index=True)

    # ----------------------------
    # 8. Keep important columns only
    # ----------------------------
    keep_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "official_split",
        "split",
        "ViewPosition",
        "path",
        "report_rel_path",
        "scene_graph_rel_path",
        "report_exists",
        "scene_graph_exists",
    ]

    subset_df = subset_df[keep_cols].copy()

    # ----------------------------
    # 9. Save outputs
    # ----------------------------
    subset_manifest_path = PROCESSED_DIR / "subset_manifest_v1.csv"
    summary_path = RESULTS_TABLES_DIR / "subset_manifest_summary.csv"

    subset_df.to_csv(subset_manifest_path, index=False)
    pd.DataFrame(subset_summary).to_csv(summary_path, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Subset manifest: {subset_manifest_path}")
    print(f"Summary table  : {summary_path}")
    print(f"Final subset rows: {len(subset_df)}")
    print("\nSubset build completed successfully.")


if __name__ == "__main__":
    main()