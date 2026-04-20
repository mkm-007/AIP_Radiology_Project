# scripts/build_clean_report_manifest.py

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TEXT_MANIFEST_PATH = OUTPUT_DIR / "report_text_manifest_v1.csv"
WORKING_TEXT_MANIFEST_PATH = OUTPUT_DIR / "working_text_manifest_v1.csv"
REPORT_TEXT_SUMMARY_PATH = TABLES_DIR / "report_text_summary_v1.csv"


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    return text


def extract_section(report_text: str, section_name: str):
    """
    Try to extract a section like FINDINGS or IMPRESSION.
    Returns None if not found.
    """
    pattern = rf"{section_name}\s*:\s*(.*?)(?=\n[A-Z ]+\s*:|\Z)"
    match = re.search(pattern, report_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def clean_for_model(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_count(text: str) -> int:
    # simple sentence estimate
    pieces = re.split(r"[.!?]+", text)
    pieces = [p.strip() for p in pieces if p.strip()]
    return len(pieces)


def main():
    print("=== BUILD CLEAN REPORT MANIFEST ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)
    print(f"Loaded working manifest rows: {len(df)}")

    rows = []

    for row in df.itertuples(index=False):
        report_path = Path(row.report_local_path)
        raw_report = read_text(report_path)
        raw_report = normalize_text(raw_report)

        findings = extract_section(raw_report, "FINDINGS")
        impression = extract_section(raw_report, "IMPRESSION")

        if findings and len(findings) >= 20:
            chosen_text = findings
            chosen_section = "findings"
        elif impression and len(impression) >= 20:
            chosen_text = impression
            chosen_section = "impression"
        else:
            chosen_text = raw_report
            chosen_section = "full_report"

        chosen_text = clean_for_model(chosen_text)

        rows.append({
            "subject_id": row.subject_id,
            "study_id": row.study_id,
            "dicom_id": row.dicom_id,
            "official_split": row.official_split,
            "report_local_path": row.report_local_path,
            "report_text_source": chosen_section,
            "report_text_clean": chosen_text,
            "report_char_count": len(chosen_text),
            "report_sentence_count": sentence_count(chosen_text),
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_TEXT_MANIFEST_PATH, index=False)

    # merge back into working manifest
    merge_cols = ["subject_id", "study_id", "dicom_id", "official_split"]
    merged_df = df.merge(report_df, on=merge_cols, how="left")
    merged_df.to_csv(WORKING_TEXT_MANIFEST_PATH, index=False)

    summary_df = pd.DataFrame([{
        "num_rows": len(report_df),
        "mean_char_count": report_df["report_char_count"].mean(),
        "median_char_count": report_df["report_char_count"].median(),
        "mean_sentence_count": report_df["report_sentence_count"].mean(),
        "median_sentence_count": report_df["report_sentence_count"].median(),
        "num_findings_used": int((report_df["report_text_source"] == "findings").sum()),
        "num_impression_used": int((report_df["report_text_source"] == "impression").sum()),
        "num_full_report_used": int((report_df["report_text_source"] == "full_report").sum()),
    }])
    summary_df.to_csv(REPORT_TEXT_SUMMARY_PATH, index=False)

    print(f"Rows written: {len(report_df)}")
    print("\nReport text source counts:")
    print(report_df["report_text_source"].value_counts())

    print("\nExample cleaned text:")
    print(report_df.iloc[0]["report_text_clean"][:400])

    print("\n=== SAVED OUTPUTS ===")
    print(f"Report text manifest : {REPORT_TEXT_MANIFEST_PATH}")
    print(f"Working text manifest: {WORKING_TEXT_MANIFEST_PATH}")
    print(f"Summary table        : {REPORT_TEXT_SUMMARY_PATH}")
    print("Clean report text manifest build completed successfully.")


if __name__ == "__main__":
    main()