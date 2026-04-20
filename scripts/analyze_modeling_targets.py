# scripts/analyze_modeling_targets.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "modeling_target_analysis_v1.csv"
TOP_TARGETS_PATH = OUTPUT_DIR / "top_region_targets_v1.csv"


def main():
    print("=== ANALYZE MODELING TARGETS ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded modeling dataset rows: {len(df)}")

    df["region_text_target"] = df["region_text_target"].fillna("").astype(str).str.strip()

    # overall stats
    total_rows = len(df)
    unique_targets = df["region_text_target"].nunique()
    duplicate_rows = total_rows - unique_targets

    print(f"Total rows                 : {total_rows}")
    print(f"Unique target texts        : {unique_targets}")
    print(f"Rows beyond unique count   : {duplicate_rows}")
    print(f"Unique/total ratio         : {unique_targets / total_rows:.4f}")

    # region-level stats
    summary_rows = []

    for region in sorted(df["canonical_region"].unique()):
        region_df = df[df["canonical_region"] == region].copy()
        n_rows = len(region_df)
        n_unique = region_df["region_text_target"].nunique()

        most_common = (
            region_df["region_text_target"]
            .value_counts()
            .head(5)
        )

        summary_rows.append({
            "canonical_region": region,
            "row_count": n_rows,
            "unique_target_count": n_unique,
            "unique_ratio": n_unique / n_rows if n_rows > 0 else 0.0,
            "top_target_frequency": int(most_common.iloc[0]) if len(most_common) > 0 else 0
        })

        print("\n" + "=" * 80)
        print(f"REGION: {region}")
        print("=" * 80)
        print(f"Rows                : {n_rows}")
        print(f"Unique target texts : {n_unique}")
        print(f"Unique ratio        : {n_unique / n_rows:.4f}")

        print("Top 5 repeated targets:")
        for text, count in most_common.items():
            print(f"  ({count}) {text}")

    summary_df = pd.DataFrame(summary_rows).sort_values("row_count", ascending=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    top_targets_df = (
        df.groupby(["canonical_region", "region_text_target"])
        .size()
        .reset_index(name="count")
        .sort_values(["canonical_region", "count"], ascending=[True, False])
    )

    # keep top 20 per region
    top_targets_df = (
        top_targets_df.groupby("canonical_region", group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_targets_df.to_csv(TOP_TARGETS_PATH, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Summary table   : {SUMMARY_PATH}")
    print(f"Top targets CSV : {TOP_TARGETS_PATH}")
    print("Modeling target analysis completed successfully.")


if __name__ == "__main__":
    main()