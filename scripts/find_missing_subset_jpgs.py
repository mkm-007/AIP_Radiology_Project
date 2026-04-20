# scripts/find_missing_subset_jpgs.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PATHS_FILE = PROJECT_ROOT / "data" / "processed" / "subset_jpg_paths.txt"
DOWNLOADED_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg_subset"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MISSING_PATHS_TXT = OUTPUT_DIR / "missing_subset_jpg_paths.txt"
SUMMARY_CSV = RESULTS_DIR / "missing_subset_jpg_summary.csv"


def normalize_rel_path(path_str: str) -> str:
    return path_str.replace("\\", "/").strip()


def main():
    print("=== FIND MISSING SUBSET JPGS ===")

    # Load expected paths
    with open(EXPECTED_PATHS_FILE, "r", encoding="utf-8") as f:
        expected_paths = [normalize_rel_path(line) for line in f if line.strip()]

    expected_set = set(expected_paths)

    # Collect downloaded jpg paths relative to the download root
    downloaded_files = list(DOWNLOADED_ROOT.rglob("*.jpg"))
    downloaded_rel_paths = []

    for p in downloaded_files:
        rel = p.relative_to(DOWNLOADED_ROOT).as_posix()
        downloaded_rel_paths.append(rel)

    downloaded_set = set(downloaded_rel_paths)

    # Find missing
    missing_paths = sorted(expected_set - downloaded_set)

    print(f"Expected JPG paths   : {len(expected_set)}")
    print(f"Downloaded JPG files : {len(downloaded_set)}")
    print(f"Missing JPG files    : {len(missing_paths)}")

    print("\nSample missing paths:")
    for p in missing_paths[:10]:
        print(f"  - {p}")

    # Save missing list
    with open(MISSING_PATHS_TXT, "w", encoding="utf-8") as f:
        for p in missing_paths:
            f.write(p + "\n")

    # Save summary
    summary_df = pd.DataFrame([{
        "expected_jpg_paths": len(expected_set),
        "downloaded_jpg_files": len(downloaded_set),
        "missing_jpg_files": len(missing_paths),
    }])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Missing paths txt: {MISSING_PATHS_TXT}")
    print(f"Summary CSV      : {SUMMARY_CSV}")
    print("Missing file detection completed successfully.")


if __name__ == "__main__":
    main()