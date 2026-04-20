# scripts/build_region_text_targets.py

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_CROP_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "region_crop_manifest_v1.csv"
REPORT_TEXT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "report_text_manifest_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "region_text_targets_v1.csv"
SUMMARY_PATH = TABLES_DIR / "region_text_target_summary_v1.csv"


REGION_KEYWORDS = {
    "left_lung": [
        "left lung", "left upper lung", "left lower lung", "left mid lung",
        "left perihilar", "left basilar", "left base", "left apical",
        "left pulmonary", "left hilar"
    ],
    "right_lung": [
        "right lung", "right upper lung", "right lower lung", "right mid lung",
        "right perihilar", "right basilar", "right base", "right apical",
        "right pulmonary", "right hilar"
    ],
    "mediastinum": [
        "mediastinum", "mediastinal", "trachea", "bronchus", "aorta"
    ],
    "heart": [
        "heart", "cardiac", "cardiomediastinal", "cardiomediastinal silhouette",
        "cardiomegaly", "cardiac silhouette"
    ],
    "left_hemidiaphragm": [
        "left hemidiaphragm", "left diaphragm", "left subdiaphragmatic"
    ],
    "right_hemidiaphragm": [
        "right hemidiaphragm", "right diaphragm", "right subdiaphragmatic"
    ]
}


def split_into_sentences(text: str):
    text = text.strip()
    if not text:
        return []

    # simple sentence split
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def find_region_sentence(report_text: str, canonical_region: str):
    sentences = split_into_sentences(report_text)
    keywords = REGION_KEYWORDS.get(canonical_region, [])

    for sent in sentences:
        sent_lower = sent.lower()
        for kw in keywords:
            if kw in sent_lower:
                return sent, True

    return "", False


def main():
    print("=== BUILD REGION TEXT TARGETS ===")

    crop_df = pd.read_csv(REGION_CROP_MANIFEST_PATH)
    report_df = pd.read_csv(REPORT_TEXT_MANIFEST_PATH)

    print(f"Loaded region crop rows : {len(crop_df)}")
    print(f"Loaded report text rows : {len(report_df)}")

    merge_cols = ["subject_id", "study_id", "dicom_id", "official_split"]
    merged_df = crop_df.merge(
        report_df[merge_cols + ["report_text_clean", "report_text_source"]],
        on=merge_cols,
        how="left"
    )

    print(f"Merged rows: {len(merged_df)}")

    target_texts = []
    has_region_text_flags = []

    for row in merged_df.itertuples(index=False):
        matched_text, matched_flag = find_region_sentence(
            report_text=row.report_text_clean,
            canonical_region=row.canonical_region
        )
        target_texts.append(matched_text)
        has_region_text_flags.append(int(matched_flag))

    merged_df["region_text_target"] = target_texts
    merged_df["has_region_text_target"] = has_region_text_flags

    merged_df.to_csv(OUTPUT_MANIFEST_PATH, index=False)

    summary_df = (
        merged_df.groupby(["official_split", "canonical_region"])
        .agg(
            total_rows=("dicom_id", "count"),
            matched_rows=("has_region_text_target", "sum")
        )
        .reset_index()
    )
    summary_df["matched_fraction"] = summary_df["matched_rows"] / summary_df["total_rows"]
    summary_df.to_csv(SUMMARY_PATH, index=False)

    total_rows = len(merged_df)
    matched_rows = int(merged_df["has_region_text_target"].sum())

    print(f"\nTotal region rows      : {total_rows}")
    print(f"Rows with text targets : {matched_rows}")
    print(f"Overall match fraction : {matched_rows / total_rows:.4f}")

    print("\nSummary by split and region:")
    print(summary_df)

    matched_examples = merged_df[merged_df["has_region_text_target"] == 1][
        ["canonical_region", "region_text_target"]
    ].head(10)

    print("\nSample matched region text targets:")
    for _, r in matched_examples.iterrows():
        print(f"- [{r['canonical_region']}] {r['region_text_target']}")

    print("\n=== SAVED OUTPUTS ===")
    print(f"Region-text manifest : {OUTPUT_MANIFEST_PATH}")
    print(f"Summary table        : {SUMMARY_PATH}")
    print("Region text target build completed successfully.")


if __name__ == "__main__":
    main()