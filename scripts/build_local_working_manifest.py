# scripts/build_local_working_manifest.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "subset_manifest_with_jpg_paths_v1.csv"

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg_subset"
REPORT_ROOT = PROJECT_ROOT / "data" / "interim" / "mimic_cxr_reports_extracted"
SCENE_GRAPH_ROOT = PROJECT_ROOT / "data" / "interim" / "chest_imagenome_scene_graphs"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_LOCAL_MANIFEST = OUTPUT_DIR / "final_local_manifest_v1.csv"
WORKING_MANIFEST = OUTPUT_DIR / "working_manifest_v1.csv"
WORKING_SUMMARY = TABLES_DIR / "working_manifest_summary.csv"

WORKING_TARGETS = {
    "train": 300,
    "valid": 60,
    "test": 60,
}

RANDOM_SEED = 42


def to_local_path(root: Path, rel_path: str) -> str:
    return str((root / rel_path).resolve())


def main():
    print("=== BUILD FINAL LOCAL + WORKING MANIFEST ===")

    df = pd.read_csv(INPUT_MANIFEST)
    print(f"Loaded resolved subset rows: {len(df)}")

    # Add local absolute paths
    df["image_local_path"] = df["jpg_rel_path"].apply(lambda x: to_local_path(IMAGE_ROOT, x))
    df["report_local_path"] = df["report_rel_path"].apply(lambda x: to_local_path(REPORT_ROOT, x))
    df["scene_graph_local_path"] = df["scene_graph_rel_path"].apply(lambda x: to_local_path(SCENE_GRAPH_ROOT, x))

    # Verify files exist
    df["image_local_exists"] = df["image_local_path"].apply(lambda x: Path(x).exists())
    df["report_local_exists"] = df["report_local_path"].apply(lambda x: Path(x).exists())
    df["scene_graph_local_exists"] = df["scene_graph_local_path"].apply(lambda x: Path(x).exists())

    before_exists = len(df)
    df = df[
        (df["image_local_exists"]) &
        (df["report_local_exists"]) &
        (df["scene_graph_local_exists"])
    ].copy()
    after_exists = len(df)

    print(f"Rows removed because some local file is missing: {before_exists - after_exists}")
    print(f"Rows remaining with all local files present: {after_exists}")

    # Save final local manifest
    df.to_csv(FINAL_LOCAL_MANIFEST, index=False)

    # Build smaller working subset for fast experimentation
    working_parts = []
    summary_rows = []

    for split_name, target_n in WORKING_TARGETS.items():
        split_df = df[df["official_split"] == split_name].copy()
        available_n = len(split_df)
        actual_n = min(target_n, available_n)

        split_df = split_df.sample(n=actual_n, random_state=RANDOM_SEED).copy()
        working_parts.append(split_df)

        summary_rows.append({
            "split": split_name,
            "available_in_final_local_manifest": available_n,
            "target_requested": target_n,
            "actual_selected": actual_n,
        })

        print(f"{split_name}: available={available_n}, selected={actual_n}")

    working_df = pd.concat(working_parts, ignore_index=True)
    working_df.to_csv(WORKING_MANIFEST, index=False)
    pd.DataFrame(summary_rows).to_csv(WORKING_SUMMARY, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Final local manifest : {FINAL_LOCAL_MANIFEST}")
    print(f"Working manifest     : {WORKING_MANIFEST}")
    print(f"Working summary      : {WORKING_SUMMARY}")
    print(f"Working subset rows  : {len(working_df)}")
    print("Manifest build completed successfully.")


if __name__ == "__main__":
    main()