# scripts/build_normalized_generation_dataset_v1.py

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET_PATH = OUTPUT_DIR / "generation_dataset_normalized_v1.csv"
SUMMARY_PATH = TABLES_DIR / "generation_dataset_normalization_summary_v1.csv"


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("___", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def normalize_heart_text(text: str) -> str:
    t = text.lower().strip()

    if "no cardiomegaly" in t:
        return "No cardiomegaly."
    if "heart size is normal" in t or "the heart size is normal" in t:
        return "Heart size is normal."
    if "cardiomediastinal silhouette is normal" in t or "the cardiomediastinal silhouette is normal" in t:
        return "Cardiomediastinal silhouette is normal."
    if "cardiomediastinal silhouette is within normal limits" in t or "the cardiomediastinal silhouette is within normal limits" in t:
        return "Cardiomediastinal silhouette is within normal limits."
    if "cardiac silhouette" in t and "normal" in t:
        return "Cardiac silhouette is normal."

    return text


def normalize_mediastinum_text(text: str) -> str:
    t = text.lower().strip()

    if "mediastinal silhouette and hilar contours are normal" in t:
        return "Mediastinal silhouette and hilar contours are normal."
    if "mediastinal and hilar contours are normal" in t:
        return "Mediastinal and hilar contours are normal."
    if "cardiomediastinal and hilar contours are within normal limits and stable" in t:
        return "Cardiomediastinal and hilar contours are within normal limits and stable."
    if "mediastinum" in t and "stable" in t:
        return "Mediastinum is stable."
    if "mediastinum" in t and "normal" in t:
        return "Mediastinum is normal."

    return text


def normalize_left_lung_text(text: str) -> str:
    t = text.lower().strip()

    if "left lung is clear" in t:
        return "Left lung is clear."
    if "left basilar atelectasis" in t or "atelectasis at the left lung base" in t or "left lung base" in t:
        return "Left basilar atelectatic change is present."
    if "left pleural effusion" in t:
        return "Left pleural effusion is present."
    if "left pneumothorax" in t:
        return "Left pneumothorax is present."

    return text


def normalize_right_lung_text(text: str) -> str:
    t = text.lower().strip()

    if "right lung is clear" in t:
        return "Right lung is clear."
    if "right basilar atelectasis" in t or "right lung base" in t or "right basilar" in t:
        return "Right basilar atelectatic or opacity change is present."
    if "right pleural effusion" in t:
        return "Right pleural effusion is present."
    if "right pneumothorax" in t:
        return "Right pneumothorax is present."

    return text


def normalize_by_region(region: str, text: str) -> str:
    text = clean_text(text)

    if region == "heart":
        text = normalize_heart_text(text)
    elif region == "mediastinum":
        text = normalize_mediastinum_text(text)
    elif region == "left_lung":
        text = normalize_left_lung_text(text)
    elif region == "right_lung":
        text = normalize_right_lung_text(text)

    text = clean_text(text)

    if text and not text.endswith((".", "!", "?")):
        text = text + "."

    return text


def main():
    print("=== BUILD NORMALIZED GENERATION DATASET V1 ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded modeling dataset rows: {len(df)}")

    original_texts = df["region_text_target"].fillna("").astype(str).tolist()
    normalized_texts = []

    for row in df.itertuples(index=False):
        normalized_texts.append(
            normalize_by_region(row.canonical_region, row.region_text_target)
        )

    df["region_text_target_original"] = df["region_text_target"]
    df["region_text_target_normalized"] = normalized_texts
    df["normalized_char_count"] = df["region_text_target_normalized"].astype(str).str.len()

    changed_mask = (
        df["region_text_target_original"].fillna("").astype(str).str.strip() !=
        df["region_text_target_normalized"].fillna("").astype(str).str.strip()
    )

    num_changed = int(changed_mask.sum())
    num_unchanged = int((~changed_mask).sum())

    df.to_csv(OUTPUT_DATASET_PATH, index=False)

    summary_rows = []

    for region in sorted(df["canonical_region"].unique()):
        sub = df[df["canonical_region"] == region].copy()
        region_changed = (
            sub["region_text_target_original"].fillna("").astype(str).str.strip() !=
            sub["region_text_target_normalized"].fillna("").astype(str).str.strip()
        ).sum()

        summary_rows.append({
            "canonical_region": region,
            "row_count": len(sub),
            "changed_rows": int(region_changed),
            "unchanged_rows": int(len(sub) - region_changed),
            "unique_original_targets": sub["region_text_target_original"].nunique(),
            "unique_normalized_targets": sub["region_text_target_normalized"].nunique(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Rows changed   : {num_changed}")
    print(f"Rows unchanged : {num_unchanged}")

    print("\nSummary by region:")
    print(summary_df)

    print("\nSample normalized targets:")
    sample_df = df[[
        "canonical_region",
        "region_text_target_original",
        "region_text_target_normalized"
    ]].head(10)

    for _, row in sample_df.iterrows():
        print(f"- [{row['canonical_region']}]")
        print(f"  original   : {row['region_text_target_original']}")
        print(f"  normalized : {row['region_text_target_normalized']}")

    print("\n=== SAVED OUTPUTS ===")
    print(f"Normalized dataset : {OUTPUT_DATASET_PATH}")
    print(f"Summary table      : {SUMMARY_PATH}")
    print("Normalized generation dataset build completed successfully.")


if __name__ == "__main__":
    main()