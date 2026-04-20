# scripts/build_final_study_reports_v1.py

from pathlib import Path
import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEN_PRED_PATH = PROJECT_ROOT / "results" / "tables" / "generation_eval_predictions_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_REPORTS_CSV = OUTPUT_DIR / "final_study_reports_v1.csv"
FINAL_REPORT_EVAL_CSV = OUTPUT_DIR / "final_study_report_eval_v1.csv"
FINAL_REPORT_SUMMARY_CSV = OUTPUT_DIR / "final_study_report_summary_v1.csv"

REGION_ORDER = ["heart", "mediastinum", "left_lung", "right_lung"]


class DummyWordNet:
    def synsets(self, word):
        return []


def normalize_sentence(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_sentences(sentences):
    seen = set()
    out = []
    for s in sentences:
        s = normalize_sentence(s)
        key = s.lower().strip()
        if not s:
            continue
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def join_sentences(sentences):
    cleaned = []
    for s in sentences:
        s = normalize_sentence(s)
        if not s:
            continue
        if s[-1] not in ".!?":
            s = s + "."
        cleaned.append(s)
    return " ".join(cleaned).strip()


def tokenize_text(text: str):
    text = str(text).strip().lower()
    return re.findall(r"[a-z0-9]+|[.,!?;:()-]", text)


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    m, n = len(ref_tokens), len(pred_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu_scores(reference: str, prediction: str):
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    if len(pred_tokens) == 0:
        return 0.0, 0.0

    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return float(bleu1), float(bleu4)


def meteor_safe(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    try:
        return float(meteor_score([ref_tokens], pred_tokens, wordnet=DummyWordNet()))
    except Exception:
        return 0.0


def exact_match(a: str, b: str) -> int:
    return int(str(a).strip().lower() == str(b).strip().lower())


def main():
    print("=== BUILD FINAL STUDY REPORTS V1 ===")

    df = pd.read_csv(GEN_PRED_PATH)
    print(f"Loaded generation prediction rows: {len(df)}")

    if df.empty:
        raise ValueError("Generation prediction file is empty.")

    # stable ordering
    df["region_order"] = df["canonical_region"].apply(
        lambda x: REGION_ORDER.index(x) if x in REGION_ORDER else 999
    )
    df = df.sort_values(["official_split", "dicom_id", "region_order"]).reset_index(drop=True)

    report_rows = []

    grouped = df.groupby(["official_split", "dicom_id"], sort=False)

    for (official_split, dicom_id), sub in grouped:
        gt_sentences = sub["ground_truth_text"].astype(str).tolist()
        gen_sentences = sub["generated_text"].astype(str).tolist()

        gt_sentences = deduplicate_sentences(gt_sentences)
        gen_sentences = deduplicate_sentences(gen_sentences)

        gt_report = join_sentences(gt_sentences)
        gen_report = join_sentences(gen_sentences)

        bleu1, bleu4 = bleu_scores(gt_report, gen_report)
        rouge_l = rouge_l_f1(gt_report, gen_report)
        meteor = meteor_safe(gt_report, gen_report)

        report_rows.append({
            "official_split": official_split,
            "dicom_id": dicom_id,
            "num_region_sentences": len(sub),
            "ground_truth_report": gt_report,
            "generated_report": gen_report,
            "exact_match": exact_match(gt_report, gen_report),
            "bleu1": bleu1,
            "bleu4": bleu4,
            "rouge_l_f1": rouge_l,
            "meteor": meteor,
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(FINAL_REPORTS_CSV, index=False)
    report_df.to_csv(FINAL_REPORT_EVAL_CSV, index=False)

    summary_df = (
        report_df.groupby("official_split")
        .agg(
            row_count=("dicom_id", "count"),
            exact_match_rate=("exact_match", "mean"),
            mean_bleu1=("bleu1", "mean"),
            mean_bleu4=("bleu4", "mean"),
            mean_rouge_l_f1=("rouge_l_f1", "mean"),
            mean_meteor=("meteor", "mean"),
            mean_num_region_sentences=("num_region_sentences", "mean"),
        )
        .reset_index()
        .sort_values("official_split")
    )
    summary_df.to_csv(FINAL_REPORT_SUMMARY_CSV, index=False)

    print("\nStudy-level summary:")
    print(summary_df)

    print("\nSample final reports:")
    print(report_df[[
        "official_split", "dicom_id", "ground_truth_report", "generated_report",
        "bleu4", "rouge_l_f1", "meteor"
    ]].head(5))

    print("\n=== SAVED OUTPUTS ===")
    print(f"Final reports CSV      : {FINAL_REPORTS_CSV}")
    print(f"Study eval CSV         : {FINAL_REPORT_EVAL_CSV}")
    print(f"Study summary CSV      : {FINAL_REPORT_SUMMARY_CSV}")
    print("Final study report build completed successfully.")


if __name__ == "__main__":
    main()