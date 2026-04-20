# scripts/build_quantitative_results_tables_v1.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "retrieval_baseline_summary_v1.csv"
TUNED_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "tuned_retrieval_summary_v1.csv"
CV_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "retrieval_cv_summary_v1.csv"
BEST_CONFIG_PATH = PROJECT_ROOT / "results" / "tables" / "retrieval_cv_best_config_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARE_TABLE_PATH = OUTPUT_DIR / "quant_results_baseline_vs_tuned_v1.csv"
CV_REPORT_TABLE_PATH = OUTPUT_DIR / "quant_results_cv_report_v1.csv"
FINAL_REPORT_TABLE_PATH = OUTPUT_DIR / "quant_results_final_report_v1.csv"


def fmt_mean_std(mean_val, std_val):
    if pd.isna(std_val):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def main():
    print("=== BUILD QUANTITATIVE RESULTS TABLES V1 ===")

    baseline_df = pd.read_csv(BASELINE_SUMMARY_PATH)
    tuned_df = pd.read_csv(TUNED_SUMMARY_PATH)
    cv_df = pd.read_csv(CV_SUMMARY_PATH)
    best_cfg_df = pd.read_csv(BEST_CONFIG_PATH)

    print(f"Loaded baseline rows : {len(baseline_df)}")
    print(f"Loaded tuned rows    : {len(tuned_df)}")
    print(f"Loaded CV rows       : {len(cv_df)}")
    print(f"Loaded best config   : {len(best_cfg_df)}")

    # --------------------------------------------------
    # 1) Baseline vs tuned comparison table
    # --------------------------------------------------
    baseline_small = baseline_df.rename(columns={
        "mean_jaccard_similarity": "baseline_mean_jaccard",
        "exact_match_rate": "baseline_exact_match",
    })[
        ["official_split", "canonical_region", "baseline_mean_jaccard", "baseline_exact_match"]
    ]

    tuned_small = tuned_df.rename(columns={
        "mean_jaccard_similarity": "tuned_mean_jaccard",
        "exact_match_rate": "tuned_exact_match",
    })[
        ["official_split", "canonical_region", "tuned_mean_jaccard", "tuned_exact_match"]
    ]

    compare_df = baseline_small.merge(
        tuned_small,
        on=["official_split", "canonical_region"],
        how="outer"
    )

    compare_df["jaccard_delta"] = (
        compare_df["tuned_mean_jaccard"] - compare_df["baseline_mean_jaccard"]
    )
    compare_df["exact_match_delta"] = (
        compare_df["tuned_exact_match"] - compare_df["baseline_exact_match"]
    )

    compare_df = compare_df.sort_values(["official_split", "canonical_region"]).reset_index(drop=True)
    compare_df.to_csv(COMPARE_TABLE_PATH, index=False)

    # --------------------------------------------------
    # 2) CV report table (mean ± std)
    # --------------------------------------------------
    cv_report_df = cv_df.copy()
    cv_report_df["jaccard_mean_std"] = cv_report_df.apply(
        lambda r: fmt_mean_std(r["mean_jaccard_mean"], r["mean_jaccard_std"]), axis=1
    )
    cv_report_df["exact_match_mean_std"] = cv_report_df.apply(
        lambda r: fmt_mean_std(r["exact_match_mean"], r["exact_match_std"]), axis=1
    )

    cv_report_df = cv_report_df[
        [
            "pca_dim",
            "metric",
            "jaccard_mean_std",
            "exact_match_mean_std",
            "mean_jaccard_mean",
            "mean_jaccard_std",
            "exact_match_mean",
            "exact_match_std",
            "total_eval_rows",
        ]
    ].sort_values("mean_jaccard_mean", ascending=False)

    cv_report_df.to_csv(CV_REPORT_TABLE_PATH, index=False)

    # --------------------------------------------------
    # 3) Final report-ready summary table
    # --------------------------------------------------
    best_row = best_cfg_df.iloc[0]

    final_rows = []

    for split_name in ["valid", "test"]:
        split_df = tuned_df[tuned_df["official_split"] == split_name].copy()

        final_rows.append({
            "section": split_name,
            "best_pca_dim": best_row["pca_dim"],
            "best_metric": best_row["metric"],
            "cv_jaccard_mean_std": fmt_mean_std(best_row["mean_jaccard_mean"], best_row["mean_jaccard_std"]),
            "cv_exact_match_mean_std": fmt_mean_std(best_row["exact_match_mean"], best_row["exact_match_std"]),
            "mean_region_jaccard": split_df["mean_jaccard_similarity"].mean(),
            "mean_region_exact_match": split_df["exact_match_rate"].mean(),
        })

    final_report_df = pd.DataFrame(final_rows)
    final_report_df.to_csv(FINAL_REPORT_TABLE_PATH, index=False)

    print("\n=== BASELINE VS TUNED ===")
    print(compare_df)

    print("\n=== CV REPORT TABLE ===")
    print(cv_report_df)

    print("\n=== FINAL REPORT TABLE ===")
    print(final_report_df)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Comparison table : {COMPARE_TABLE_PATH}")
    print(f"CV report table  : {CV_REPORT_TABLE_PATH}")
    print(f"Final report tbl : {FINAL_REPORT_TABLE_PATH}")
    print("Quantitative results table build completed successfully.")


if __name__ == "__main__":
    main()