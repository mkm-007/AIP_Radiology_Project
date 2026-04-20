# scripts/inspect_region_names.py

from pathlib import Path
import pandas as pd
import json
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

REGION_COUNTS_CSV = OUTPUT_TABLE_DIR / "region_name_counts.csv"
REGION_PREVIEW_CSV = OUTPUT_TABLE_DIR / "region_name_preview.csv"


def load_scene_graph(scene_graph_path: Path):
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_region_names(scene_graph: dict):
    """
    Try to collect region/object names from the scene graph.
    This is written defensively because different JSON entries may store names slightly differently.
    """
    names = []
    objects = scene_graph.get("objects", [])

    for obj in objects:
        candidate_names = []

        if isinstance(obj, dict):
            # Common possibilities
            for key in ["name", "label", "bbox_name", "object_name"]:
                if key in obj and isinstance(obj[key], str):
                    candidate_names.append(obj[key])

            # Sometimes there may be nested strings or lists
            for key in ["names", "labels"]:
                if key in obj and isinstance(obj[key], list):
                    candidate_names.extend([x for x in obj[key] if isinstance(x, str)])

        # Keep only unique candidates for this object
        for cname in candidate_names:
            cleaned = cname.strip().lower()
            if cleaned:
                names.append(cleaned)

    return names


def main():
    print("=== INSPECT REGION NAMES ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)
    print(f"Loaded working manifest rows: {len(df)}")

    region_counter = Counter()
    preview_rows = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        scene_graph_path = Path(row.scene_graph_local_path)
        scene_graph = load_scene_graph(scene_graph_path)

        region_names = extract_region_names(scene_graph)

        # Update global counter
        region_counter.update(region_names)

        # Save a small preview for the first few samples
        if idx <= 10:
            preview_rows.append({
                "sample_index": idx,
                "dicom_id": row.dicom_id,
                "official_split": row.official_split,
                "region_names_found": ", ".join(region_names[:20])
            })

    # Build count dataframe
    count_df = pd.DataFrame(region_counter.items(), columns=["region_name", "count"])
    count_df = count_df.sort_values(by="count", ascending=False).reset_index(drop=True)

    preview_df = pd.DataFrame(preview_rows)

    count_df.to_csv(REGION_COUNTS_CSV, index=False)
    preview_df.to_csv(REGION_PREVIEW_CSV, index=False)

    print(f"Unique region names found: {len(count_df)}")
    print("\nTop 30 region names:")
    print(count_df.head(30))

    print("\nSaved files:")
    print(f"  - {REGION_COUNTS_CSV}")
    print(f"  - {REGION_PREVIEW_CSV}")
    print("Region inspection completed successfully.")


if __name__ == "__main__":
    main()