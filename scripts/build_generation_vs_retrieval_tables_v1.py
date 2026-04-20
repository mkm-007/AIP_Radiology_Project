# scripts/build_generation_vs_retrieval_tables_v1.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "tuned_retrieval_summary_v1.csv"
GEN_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "generation_eval_summary_v1.csv"
GEN_OVERALL_PATH = PROJECT_ROOT / "results" / "tables" / "generation_eval_overall_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_REGION_COMPARE_PATH = OUTPUT_DIR / "generation_vs_retrieval_per_region_v1.csv"
OVERALL_COMPARE_PATH = OUTPUT_DIR / "generation_vs_retrieval_overall_v1.csv"
REPORT_READY_PATH = OUTPUT_DIR / "generation_vs_retrieval_report_table_v1.csv"


def main():
    print("=== BUILD GENERATION VS RETRIEVAL TABLES V1 ===")

    retrieval_df = pd.read_csv(RETRIEVAL_SUMMARY_PATH)
    gen_df = pd.read_csv(GEN_SUMMARY_PATH)
    gen_overall_df = pd.read_csv(GEN_OVERALL_PATH)

    print(f"Loaded retrieval rows : {len(retrieval_df)}")
    print(f"Loaded generation rows: {len(gen_df)}")
    print(f"Loaded generation overall rows: {len(gen_overall_df)}")

    # -----------------------------
    # Per-region comparison
    # -----------------------------
    retrieval_small = retrieval_df.rename(columns={
        "mean_jaccard_similarity": "retrieval_mean_jaccard",
        "exact_match_rate": "retrieval_exact_match_rate"
    })[
        ["official_split", "canonical_region", "row_count",
         "retrieval_mean_jaccard", "retrieval_exact_match_rate"]
    ]

    gen_small = gen_df.rename(columns={
        "exact_match_rate": "generation_exact_match_rate",
        "mean_bleu1": "generation_mean_bleu1",
        "mean_bleu4": "generation_mean_bleu4",
        "mean_rouge_l_f1": "generation_mean_rouge_l_f1",
        "mean_meteor": "generation_mean_meteor"
    })[
        ["official_split", "canonical_region",
         "generation_exact_match_rate",
         "generation_mean_bleu1",
         "generation_mean_bleu4",
         "generation_mean_rouge_l_f1",
         "generation_mean_meteor"]
    ]

    per_region_df = retrieval_small.merge(
        gen_small,
        on=["official_split", "canonical_region"],
        how="outer"
    )

    per_region_df["exact_match_delta_gen_minus_ret"] = (
        per_region_df["generation_exact_match_rate"] - per_region_df["retrieval_exact_match_rate"]
    )

    per_region_df = per_region_df.sort_values(
        ["official_split", "canonical_region"]
    ).reset_index(drop=True)

    per_region_df.to_csv(PER_REGION_COMPARE_PATH, index=False)

    # -----------------------------
    # Overall comparison by split
    # -----------------------------
    retrieval_overall_df = (
        retrieval_df.groupby("official_split")
        .agg(
            retrieval_mean_jaccard=("mean_jaccard_similarity", "mean"),
            retrieval_exact_match_rate=("exact_match_rate", "mean")
        )
        .reset_index()
    )

    gen_overall_small = gen_overall_df.rename(columns={
        "exact_match_rate": "generation_exact_match_rate",
        "mean_bleu1": "generation_mean_bleu1",
        "mean_bleu4": "generation_mean_bleu4",
        "mean_rouge_l_f1": "generation_mean_rouge_l_f1",
        "mean_meteor": "generation_mean_meteor"
    })

    overall_df = retrieval_overall_df.merge(
        gen_overall_small,
        on="official_split",
        how="outer"
    )

    overall_df["exact_match_delta_gen_minus_ret"] = (
        overall_df["generation_exact_match_rate"] - overall_df["retrieval_exact_match_rate"]
    )

    overall_df = overall_df.sort_values("official_split").reset_index(drop=True)
    overall_df.to_csv(OVERALL_COMPARE_PATH, index=False)

    # -----------------------------
    # Report-ready compact table
    # -----------------------------
    report_rows = []

    for _, row in overall_df.iterrows():
        report_rows.append({
            "split": row["official_split"],
            "retrieval_jaccard": row["retrieval_mean_jaccard"],
            "retrieval_exact_match": row["retrieval_exact_match_rate"],
            "generation_exact_match": row["generation_exact_match_rate"],
            "generation_bleu1": row["generation_mean_bleu1"],
            "generation_bleu4": row["generation_mean_bleu4"],
            "generation_rouge_l_f1": row["generation_mean_rouge_l_f1"],
            "generation_meteor": row["generation_mean_meteor"],
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_READY_PATH, index=False)

    print("\n=== PER-REGION COMPARISON ===")
    print(per_region_df)

    print("\n=== OVERALL COMPARISON ===")
    print(overall_df)

    print("\n=== REPORT-READY TABLE ===")
    print(report_df)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Per-region table : {PER_REGION_COMPARE_PATH}")
    print(f"Overall table    : {OVERALL_COMPARE_PATH}")
    print(f"Report table     : {REPORT_READY_PATH}")
    print("Generation vs retrieval table build completed successfully.")


if __name__ == "__main__":
    main()