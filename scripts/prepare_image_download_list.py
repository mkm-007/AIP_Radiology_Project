# scripts/prepare_image_download_list.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "subset_manifest_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_LIST_PATH = OUTPUT_DIR / "subset_image_paths.txt"
PREVIEW_CSV_PATH = OUTPUT_DIR / "subset_image_path_preview.csv"

def main():
    print("=== PREPARE IMAGE DOWNLOAD LIST ===")

    df = pd.read_csv(SUBSET_MANIFEST_PATH)

    print(f"Loaded subset manifest rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "path" not in df.columns:
        raise ValueError("Column 'path' not found in subset manifest.")

    # Keep only rows with non-null path
    before = len(df)
    df = df[df["path"].notna()].copy()
    after = len(df)

    print(f"Rows removed due to missing path: {before - after}")
    print(f"Rows remaining with valid path: {after}")

    # Normalize slashes
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)

    # Remove duplicate image paths if any
    unique_paths = df["path"].drop_duplicates().tolist()

    print(f"Unique image paths: {len(unique_paths)}")

    print("\nSample image paths:")
    for p in unique_paths[:10]:
        print(f"  - {p}")

    # Save as plain text list (one path per line)
    with open(DOWNLOAD_LIST_PATH, "w", encoding="utf-8") as f:
        for p in unique_paths:
            f.write(p + "\n")

    # Save preview CSV for easy inspection
    df[["subject_id", "study_id", "dicom_id", "official_split", "path"]].head(50).to_csv(
        PREVIEW_CSV_PATH, index=False
    )

    print("\n=== SAVED OUTPUTS ===")
    print(f"Download list: {DOWNLOAD_LIST_PATH}")
    print(f"Preview CSV  : {PREVIEW_CSV_PATH}")
    print("Image download list preparation completed successfully.")


if __name__ == "__main__":
    main()