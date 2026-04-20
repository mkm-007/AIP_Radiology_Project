# scripts/extract_archives.py

from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_ZIP = PROJECT_ROOT / "data" / "raw" / "mimic_cxr_reports" / "mimic-cxr-reports.zip"
SCENE_GRAPH_ZIP = PROJECT_ROOT / "data" / "raw" / "chest_imagenome" / "scene_graph.zip"

REPORTS_OUT = PROJECT_ROOT / "data" / "interim" / "mimic_cxr_reports_extracted"
SCENE_GRAPH_OUT = PROJECT_ROOT / "data" / "interim" / "chest_imagenome_scene_graphs"


def extract_zip(zip_path: Path, output_dir: Path, label: str):
    print(f"\n=== Extracting {label} ===")
    print(f"ZIP file   : {zip_path}")
    print(f"Output dir : {output_dir}")

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing file: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"Files inside zip: {len(members)}")

        # Extract all
        zf.extractall(output_dir)

    print(f"Finished extracting {label}.")


def count_files(folder: Path, suffix: str = None):
    if not folder.exists():
        return 0

    if suffix is None:
        return sum(1 for p in folder.rglob("*") if p.is_file())

    return sum(1 for p in folder.rglob(f"*{suffix}") if p.is_file())


def main():
    print("=== ARCHIVE EXTRACTION START ===")

    extract_zip(REPORTS_ZIP, REPORTS_OUT, "MIMIC-CXR reports")
    extract_zip(SCENE_GRAPH_ZIP, SCENE_GRAPH_OUT, "Chest ImaGenome scene graphs")

    report_txt_count = count_files(REPORTS_OUT, ".txt")
    scene_json_count = count_files(SCENE_GRAPH_OUT, ".json")

    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Extracted report .txt files : {report_txt_count}")
    print(f"Extracted scene graph .json : {scene_json_count}")
    print("Extraction completed successfully.")


if __name__ == "__main__":
    main()