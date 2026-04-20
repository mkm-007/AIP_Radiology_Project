# scripts/inspect_structured_data.py

from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIMIC_JPG_DIR = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_jpg"
CHEST_IMAGENOME_DIR = PROJECT_ROOT / "data" / "raw" / "chest_imagenome"

REPORTS_EXTRACTED_DIR = PROJECT_ROOT / "data" / "interim" / "mimic_cxr_reports_extracted"
SCENE_GRAPH_EXTRACTED_DIR = PROJECT_ROOT / "data" / "interim" / "chest_imagenome_scene_graphs"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = MIMIC_JPG_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"
SPLIT_PATH = MIMIC_JPG_DIR / "mimic-cxr-2.0.0-split.csv.gz"

TRAIN_PATH = CHEST_IMAGENOME_DIR / "train.csv"
VALID_PATH = CHEST_IMAGENOME_DIR / "valid.csv"
TEST_PATH = CHEST_IMAGENOME_DIR / "test.csv"
IMAGES_TO_AVOID_PATH = CHEST_IMAGENOME_DIR / "images_to_avoid.csv"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def list_sample_files(root_dir: Path, suffix: str, n: int = 5):
    files = sorted(root_dir.rglob(f"*{suffix}"))
    return files[:n], len(files)


def main():
    inspection_summary = {}

    print_section("1. LOAD MIMIC-CXR-JPG METADATA")
    metadata = pd.read_csv(METADATA_PATH, compression="gzip")
    print(f"Metadata shape: {metadata.shape}")
    print("Metadata columns:")
    print(list(metadata.columns))
    print("\nFirst 5 rows:")
    print(metadata.head())

    inspection_summary["metadata_shape"] = metadata.shape
    inspection_summary["metadata_columns"] = list(metadata.columns)

    print_section("2. LOAD MIMIC-CXR-JPG SPLIT FILE")
    split_df = pd.read_csv(SPLIT_PATH, compression="gzip")
    print(f"Split shape: {split_df.shape}")
    print("Split columns:")
    print(list(split_df.columns))
    print("\nSplit value counts:")
    if "split" in split_df.columns:
        print(split_df["split"].value_counts())
    else:
        print("No 'split' column found.")
    print("\nFirst 5 rows:")
    print(split_df.head())

    inspection_summary["split_shape"] = split_df.shape
    inspection_summary["split_columns"] = list(split_df.columns)

    print_section("3. LOAD CHEST IMAGENOME SPLITS")
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    avoid_df = pd.read_csv(IMAGES_TO_AVOID_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"Images-to-avoid shape: {avoid_df.shape}")

    print("\nTrain columns:")
    print(list(train_df.columns))
    print("\nValid columns:")
    print(list(valid_df.columns))
    print("\nTest columns:")
    print(list(test_df.columns))
    print("\nImages-to-avoid columns:")
    print(list(avoid_df.columns))

    inspection_summary["train_shape"] = train_df.shape
    inspection_summary["valid_shape"] = valid_df.shape
    inspection_summary["test_shape"] = test_df.shape
    inspection_summary["avoid_shape"] = avoid_df.shape

    print_section("4. SAMPLE REPORT FILES")
    sample_reports, num_reports = list_sample_files(REPORTS_EXTRACTED_DIR, ".txt", n=5)
    print(f"Total report .txt files found: {num_reports}")
    print("Sample report paths:")
    for p in sample_reports:
        print(f"  - {p.relative_to(REPORTS_EXTRACTED_DIR)}")

    inspection_summary["num_report_txt_files"] = num_reports
    inspection_summary["sample_report_paths"] = [str(p.relative_to(REPORTS_EXTRACTED_DIR)) for p in sample_reports]

    print_section("5. SAMPLE SCENE GRAPH FILES")
    sample_scene_graphs, num_scene_graphs = list_sample_files(SCENE_GRAPH_EXTRACTED_DIR, ".json", n=5)
    print(f"Total scene graph .json files found: {num_scene_graphs}")
    print("Sample scene graph paths:")
    for p in sample_scene_graphs:
        print(f"  - {p.relative_to(SCENE_GRAPH_EXTRACTED_DIR)}")

    inspection_summary["num_scene_graph_json_files"] = num_scene_graphs
    inspection_summary["sample_scene_graph_paths"] = [str(p.relative_to(SCENE_GRAPH_EXTRACTED_DIR)) for p in sample_scene_graphs]

    print_section("6. OPEN ONE SAMPLE SCENE GRAPH JSON")
    if sample_scene_graphs:
        with open(sample_scene_graphs[0], "r", encoding="utf-8") as f:
            sample_json = json.load(f)

        print(f"Sample scene graph file: {sample_scene_graphs[0].name}")
        print("Top-level keys:")
        print(list(sample_json.keys()))

        inspection_summary["sample_scene_graph_top_keys"] = list(sample_json.keys())
    else:
        print("No scene graph JSON files found.")
        inspection_summary["sample_scene_graph_top_keys"] = []

    print_section("7. SAVE SUMMARY")
    summary_path = OUTPUT_DIR / "dataset_inspection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(inspection_summary, f, indent=2)

    print(f"Saved summary to: {summary_path}")
    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()