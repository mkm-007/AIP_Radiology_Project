# scripts/build_qualitative_comparison_figures_v1.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEN_PATH = PROJECT_ROOT / "results" / "tables" / "generation_eval_predictions_v1.csv"
RET_PATH = PROJECT_ROOT / "results" / "tables" / "tuned_retrieval_predictions_v1.csv"

OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_FIG_DIR = PROJECT_ROOT / "results" / "figures" / "qualitative_examples"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV_PATH = OUTPUT_TABLE_DIR / "qualitative_examples_merged_v1.csv"
SELECTED_CSV_PATH = OUTPUT_TABLE_DIR / "qualitative_examples_selected_v1.csv"

NUM_TOP = 6
NUM_HARD = 4


def wrap_text(text, width=60):
    return "\n".join(textwrap.wrap(str(text), width=width))


def build_panel(row, save_path):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axis("off")

    header = (
        f"Split: {row['official_split']} | "
        f"Region: {row['canonical_region']} | "
        f"DICOM: {row['dicom_id']}"
    )

    metrics = (
        f"Generation -> BLEU-1: {row['bleu1']:.3f}, BLEU-4: {row['bleu4']:.3f}, "
        f"ROUGE-L: {row['rouge_l_f1']:.3f}, METEOR: {row['meteor']:.3f}, "
        f"Exact Match: {int(row['generation_exact_match'])}\n"
        f"Retrieval  -> Jaccard: {row['retrieval_jaccard_similarity']:.3f}, "
        f"Exact Match: {int(row['retrieval_exact_match'])}"
    )

    gt = "GROUND TRUTH:\n" + wrap_text(row["ground_truth_text"], width=70)
    ret = "RETRIEVAL:\n" + wrap_text(row["retrieval_text"], width=70)
    gen = "GENERATION:\n" + wrap_text(row["generation_text"], width=70)

    full_text = (
        header + "\n\n" +
        metrics + "\n\n" +
        gt + "\n\n" +
        ret + "\n\n" +
        gen
    )

    ax.text(
        0.01, 0.99, full_text,
        va="top", ha="left",
        fontsize=10,
        family="monospace"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== BUILD QUALITATIVE COMPARISON FIGURES V1 ===")

    gen_df = pd.read_csv(GEN_PATH)
    ret_df = pd.read_csv(RET_PATH)

    print(f"Loaded generation rows: {len(gen_df)}")
    print(f"Loaded retrieval rows : {len(ret_df)}")

    gen_small = gen_df.rename(columns={
        "generated_text": "generation_text",
        "exact_match": "generation_exact_match"
    })[
        [
            "official_split", "canonical_region", "dicom_id",
            "ground_truth_text", "generation_text",
            "generation_exact_match", "bleu1", "bleu4", "rouge_l_f1", "meteor"
        ]
    ]

    ret_small = ret_df.rename(columns={
        "predicted_text": "retrieval_text",
        "exact_match": "retrieval_exact_match",
        "jaccard_similarity": "retrieval_jaccard_similarity",
        "query_dicom_id": "dicom_id"
    })[
        [
            "official_split", "canonical_region", "dicom_id",
            "retrieval_text", "retrieval_exact_match", "retrieval_jaccard_similarity"
        ]
    ]

    merged = gen_small.merge(
        ret_small,
        on=["official_split", "canonical_region", "dicom_id"],
        how="inner"
    )

    if merged.empty:
        raise ValueError("Merged qualitative dataframe is empty.")

    merged["gen_minus_ret_exact"] = (
        merged["generation_exact_match"] - merged["retrieval_exact_match"]
    )
    merged["score_for_sort"] = (
        merged["bleu4"] + merged["rouge_l_f1"] + merged["meteor"]
    )

    merged.to_csv(MERGED_CSV_PATH, index=False)

    # Pick strong examples from test first, then valid
    top_df = (
        merged.sort_values(
            by=["official_split", "score_for_sort"],
            ascending=[True, False]
        )
        .query("official_split == 'test'")
        .head(NUM_TOP)
        .copy()
    )

    # Pick hard examples with low generation quality
    hard_df = (
        merged.sort_values(
            by=["score_for_sort", "official_split"],
            ascending=[True, True]
        )
        .head(NUM_HARD)
        .copy()
    )

    selected = pd.concat([top_df, hard_df], ignore_index=True).drop_duplicates(
        subset=["official_split", "canonical_region", "dicom_id"]
    ).reset_index(drop=True)

    selected.to_csv(SELECTED_CSV_PATH, index=False)

    print(f"Merged rows   : {len(merged)}")
    print(f"Selected rows : {len(selected)}")

    for i, row in selected.iterrows():
        out_path = OUTPUT_FIG_DIR / f"qual_example_{i+1}_{row['official_split']}_{row['canonical_region']}_{row['dicom_id']}.png"
        build_panel(row, out_path)

    print("\nSelected examples:")
    print(selected[[
        "official_split", "canonical_region", "dicom_id",
        "bleu4", "rouge_l_f1", "meteor",
        "retrieval_jaccard_similarity"
    ]])

    print("\n=== SAVED OUTPUTS ===")
    print(f"Merged CSV   : {MERGED_CSV_PATH}")
    print(f"Selected CSV : {SELECTED_CSV_PATH}")
    print(f"Figure dir   : {OUTPUT_FIG_DIR}")
    print("Qualitative comparison figure build completed successfully.")


if __name__ == "__main__":
    main()