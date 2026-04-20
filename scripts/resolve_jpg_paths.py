# scripts/resolve_jpg_paths.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "subset_manifest_v1.csv"
IMAGE_FILENAMES_PATH = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg" / "IMAGE_FILENAMES"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESOLVED_MANIFEST_PATH = OUTPUT_DIR / "subset_manifest_with_jpg_paths_v1.csv"
JPG_PATHS_TXT_PATH = OUTPUT_DIR / "subset_jpg_paths.txt"
SUMMARY_PATH = RESULTS_DIR / "subset_jpg_resolution_summary.csv"


def extract_dicom_id_from_image_path(path_str: str) -> str:
    """
    Example input:
    files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg

    Returns:
    02aa804e-bde0afdd-112c0b34-7bc16630-4e384014
    """
    return Path(path_str).stem


def main():
    print("=== RESOLVE JPG PATHS ===")

    # Load subset manifest
    subset_df = pd.read_csv(SUBSET_MANIFEST_PATH)
    print(f"Loaded subset manifest rows: {len(subset_df)}")

    if "dicom_id" not in subset_df.columns:
        raise ValueError("subset manifest missing 'dicom_id' column")

    # Load IMAGE_FILENAMES text file
    with open(IMAGE_FILENAMES_PATH, "r", encoding="utf-8") as f:
        image_paths = [line.strip().replace("\\", "/") for line in f if line.strip()]

    print(f"Loaded IMAGE_FILENAMES entries: {len(image_paths)}")

    image_df = pd.DataFrame({"jpg_rel_path": image_paths})
    image_df["dicom_id"] = image_df["jpg_rel_path"].apply(extract_dicom_id_from_image_path)

    # Merge by dicom_id
    merged_df = subset_df.merge(
        image_df,
        on="dicom_id",
        how="left"
    )

    matched = merged_df["jpg_rel_path"].notna().sum()
    unmatched = merged_df["jpg_rel_path"].isna().sum()

    print(f"Matched rows to JPG paths   : {matched}")
    print(f"Unmatched rows              : {unmatched}")

    # Save summary
    summary_df = pd.DataFrame([{
        "subset_rows": len(subset_df),
        "image_filenames_entries": len(image_paths),
        "matched_rows": int(matched),
        "unmatched_rows": int(unmatched),
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    # Keep only matched rows for download file
    matched_df = merged_df[merged_df["jpg_rel_path"].notna()].copy()

    # Save resolved manifest
    matched_df.to_csv(RESOLVED_MANIFEST_PATH, index=False)

    # Save JPG paths text file (one path per line)
    unique_jpg_paths = matched_df["jpg_rel_path"].drop_duplicates().tolist()
    with open(JPG_PATHS_TXT_PATH, "w", encoding="utf-8") as f:
        for p in unique_jpg_paths:
            f.write(p + "\n")

    print("\nSample resolved JPG paths:")
    for p in unique_jpg_paths[:10]:
        print(f"  - {p}")

    print("\n=== SAVED OUTPUTS ===")
    print(f"Resolved manifest : {RESOLVED_MANIFEST_PATH}")
    print(f"JPG paths txt     : {JPG_PATHS_TXT_PATH}")
    print(f"Summary CSV       : {SUMMARY_PATH}")
    print("JPG path resolution completed successfully.")


if __name__ == "__main__":
    main()