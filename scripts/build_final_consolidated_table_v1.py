from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T = PROJECT_ROOT / "results" / "tables"

ret_cv   = pd.read_csv(T / "retrieval_cv_best_config_v1.csv")
gen_cv   = pd.read_csv(T / "generation_cv_best_config_v1.csv")
ret_eval = pd.read_csv(T / "tuned_retrieval_summary_v1.csv")
gen_eval = pd.read_csv(T / "generation_eval_overall_v1.csv")
study    = pd.read_csv(T / "final_study_report_summary_v1.csv")

rows = []
for split in ["valid", "test"]:
    r = ret_eval[ret_eval["official_split"]==split]
    g = gen_eval[gen_eval["official_split"]==split]
    s = study[study["official_split"]==split]
    rows.append({
        "split": split,
        "retrieval_CV_jaccard": f"{ret_cv.iloc[0]['mean_jaccard_mean']:.4f} ± {ret_cv.iloc[0]['mean_jaccard_std']:.4f}",
        "retrieval_jaccard": round(float(r["mean_jaccard_similarity"].mean()), 4),
        "retrieval_exact_match": round(float(r["exact_match_rate"].mean()), 4),
        "gen_CV_bleu1": f"{gen_cv.iloc[0]['bleu1_mean']:.4f} ± {gen_cv.iloc[0]['bleu1_std']:.4f}",
        "gen_CV_rouge_l": f"{gen_cv.iloc[0]['rouge_l_mean']:.4f} ± {gen_cv.iloc[0]['rouge_l_std']:.4f}",
        "gen_bleu1": round(float(g["mean_bleu1"].values[0]), 4),
        "gen_bleu4": round(float(g["mean_bleu4"].values[0]), 4),
        "gen_rouge_l": round(float(g["mean_rouge_l_f1"].values[0]), 4),
        "gen_meteor": round(float(g["mean_meteor"].values[0]), 4),
        "study_bleu1": round(float(s["mean_bleu1"].values[0]), 4),
        "study_rouge_l": round(float(s["mean_rouge_l_f1"].values[0]), 4),
        "study_meteor": round(float(s["mean_meteor"].values[0]), 4),
    })

df = pd.DataFrame(rows)
df.to_csv(T / "final_consolidated_results_v1.csv", index=False)
print(df.to_string(index=False))
print(f"\nSaved: {T / 'final_consolidated_results_v1.csv'}")