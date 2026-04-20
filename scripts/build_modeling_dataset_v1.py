# scripts/build_modeling_dataset_v1.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "region_text_targets_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET_PATH = OUTPUT_DIR / "modeling_dataset_v1.csv"
SUMMARY_PATH = TABLES_DIR / "modeling_dataset_summary_v1.csv"

KEEP_REGIONS = [
    "heart",
    "mediastinum",
    "left_lung",
    "right_lung"
]

MIN_TEXT_CHARS = 10


def main():
    print("=== BUILD MODELING DATASET V1 ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded region-text rows: {len(df)}")

    # Keep only target regions
    before_region_filter = len(df)
    df = df[df["canonical_region"].isin(KEEP_REGIONS)].copy()
    after_region_filter = len(df)

    print(f"Rows removed by region filter: {before_region_filter - after_region_filter}")
    print(f"Rows remaining after region filter: {after_region_filter}")

    # Keep only rows with matched text targets
    before_match_filter = len(df)
    df = df[df["has_region_text_target"] == 1].copy()
    after_match_filter = len(df)

    print(f"Rows removed with no text target: {before_match_filter - after_match_filter}")
    print(f"Rows remaining after text-target filter: {after_match_filter}")

    # Keep only rows with non-trivial target text
    df["region_text_target"] = df["region_text_target"].fillna("").astype(str).str.strip()
    df["target_char_count"] = df["region_text_target"].str.len()

    before_textlen_filter = len(df)
    df = df[df["target_char_count"] >= MIN_TEXT_CHARS].copy()
    after_textlen_filter = len(df)

    print(f"Rows removed by min text length filter: {before_textlen_filter - after_textlen_filter}")
    print(f"Rows remaining after text length filter: {after_textlen_filter}")

    # Save modeling dataset
    df.to_csv(OUTPUT_DATASET_PATH, index=False)

    # Build summary table
    summary_df = (
        df.groupby(["official_split", "canonical_region"])
        .agg(
            row_count=("dicom_id", "count"),
            mean_target_char_count=("target_char_count", "mean")
        )
        .reset_index()
        .sort_values(["official_split", "canonical_region"])
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nFinal modeling dataset rows:", len(df))
    print("\nSummary by split and region:")
    print(summary_df)

    print("\nExample targets:")
    for _, row in df[["canonical_region", "region_text_target"]].head(10).iterrows():
        print(f"- [{row['canonical_region']}] {row['region_text_target']}")

    print("\n=== SAVED OUTPUTS ===")
    print(f"Modeling dataset : {OUTPUT_DATASET_PATH}")
    print(f"Summary table     : {SUMMARY_PATH}")
    print("Modeling dataset build completed successfully.")


if __name__ == "__main__":
    main()