# scripts/build_region_manifest.py

from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"
REGION_CONFIG_PATH = PROJECT_ROOT / "configs" / "region_config_v1.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

REGION_MANIFEST_PATH = OUTPUT_DIR / "region_manifest_v1.csv"
REGION_SUMMARY_PATH = TABLES_DIR / "region_manifest_summary_v1.csv"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_candidate_names(obj: dict):
    names = []

    for key in ["name", "label", "bbox_name", "object_name"]:
        if key in obj and isinstance(obj[key], str):
            names.append(obj[key].strip().lower())

    for key in ["names", "labels"]:
        if key in obj and isinstance(obj[key], list):
            for x in obj[key]:
                if isinstance(x, str):
                    names.append(x.strip().lower())

    # preserve order, remove duplicates
    return list(dict.fromkeys([n for n in names if n]))


def find_canonical_region_matches(scene_graph: dict, raw_to_canonical: dict):
    """
    Returns one bbox object per canonical region if found.
    We keep the FIRST matching object for each canonical region.
    """
    objects = scene_graph.get("objects", [])
    matched = {}

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        candidate_names = extract_candidate_names(obj)

        canonical_hits = []
        for raw_name in candidate_names:
            if raw_name in raw_to_canonical:
                canonical_hits.append(raw_to_canonical[raw_name])

        for canonical_region in canonical_hits:
            if canonical_region not in matched:
                matched[canonical_region] = obj

    return matched


def main():
    print("=== BUILD REGION MANIFEST ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)

    with open(REGION_CONFIG_PATH, "r", encoding="utf-8") as f:
        region_config = json.load(f)

    canonical_regions = region_config["canonical_regions"]
    raw_to_canonical = {
        k.strip().lower(): v for k, v in region_config["raw_to_canonical"].items()
    }

    rows = []

    for row in df.itertuples(index=False):
        scene_graph = load_json(Path(row.scene_graph_local_path))
        matched_regions = find_canonical_region_matches(scene_graph, raw_to_canonical)

        for canonical_region in canonical_regions:
            if canonical_region not in matched_regions:
                continue

            obj = matched_regions[canonical_region]

            rows.append({
                "subject_id": row.subject_id,
                "study_id": row.study_id,
                "dicom_id": row.dicom_id,
                "official_split": row.official_split,
                "view_position": row.ViewPosition,

                "image_local_path": row.image_local_path,
                "report_local_path": row.report_local_path,
                "scene_graph_local_path": row.scene_graph_local_path,

                "canonical_region": canonical_region,

                # resized bbox
                "x1": obj.get("x1"),
                "y1": obj.get("y1"),
                "x2": obj.get("x2"),
                "y2": obj.get("y2"),
                "width": obj.get("width"),
                "height": obj.get("height"),

                # original-image bbox (this is what we plan to use later)
                "original_x1": obj.get("original_x1"),
                "original_y1": obj.get("original_y1"),
                "original_x2": obj.get("original_x2"),
                "original_y2": obj.get("original_y2"),
                "original_width": obj.get("original_width"),
                "original_height": obj.get("original_height"),

                "bbox_name": obj.get("bbox_name"),
                "object_name": obj.get("name"),
                "object_id": obj.get("object_id")
            })

    region_df = pd.DataFrame(rows)

    print(f"Region-level rows created: {len(region_df)}")

    # Save full region-level manifest
    region_df.to_csv(REGION_MANIFEST_PATH, index=False)

    # Build summary
    summary_df = (
        region_df.groupby(["official_split", "canonical_region"])
        .size()
        .reset_index(name="row_count")
        .sort_values(["official_split", "canonical_region"])
    )
    summary_df.to_csv(REGION_SUMMARY_PATH, index=False)

    print("\nRegion counts by split and canonical region:")
    print(summary_df)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Region manifest : {REGION_MANIFEST_PATH}")
    print(f"Summary table   : {REGION_SUMMARY_PATH}")
    print("Region manifest build completed successfully.")


if __name__ == "__main__":
    main()