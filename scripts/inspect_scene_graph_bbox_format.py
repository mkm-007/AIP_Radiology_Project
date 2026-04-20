# scripts/inspect_scene_graph_bbox_format.py

from pathlib import Path
import pandas as pd
import json
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"
REGION_CONFIG_PATH = PROJECT_ROOT / "configs" / "region_config_v1.json"


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

    return list(dict.fromkeys([n for n in names if n]))


def main():
    print("=== INSPECT SCENE GRAPH BBOX FORMAT ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)

    with open(REGION_CONFIG_PATH, "r", encoding="utf-8") as f:
        region_config = json.load(f)

    raw_to_canonical = {
        k.strip().lower(): v for k, v in region_config["raw_to_canonical"].items()
    }

    # Use the first sample only for deep inspection
    row = df.iloc[0]
    scene_graph_path = Path(row["scene_graph_local_path"])

    print(f"Using sample dicom_id: {row['dicom_id']}")
    print(f"Scene graph path     : {scene_graph_path}")

    sg = load_json(scene_graph_path)

    print("\nTop-level keys:")
    print(list(sg.keys()))

    objects = sg.get("objects", [])
    print(f"\nNumber of objects in scene graph: {len(objects)}")

    if not objects:
        print("No objects found. Stopping.")
        return

    print("\nKeys of the first object:")
    print(list(objects[0].keys()))

    print("\nFirst object full content:")
    pprint(objects[0], sort_dicts=False)

    print("\n" + "=" * 80)
    print("MATCHES TO CANONICAL REGIONS")
    print("=" * 80)

    matched_count = 0

    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue

        candidate_names = extract_candidate_names(obj)

        canonical_matches = []
        for raw_name in candidate_names:
            if raw_name in raw_to_canonical:
                canonical_matches.append((raw_name, raw_to_canonical[raw_name]))

        if canonical_matches:
            matched_count += 1
            print(f"\nObject index: {idx}")
            print(f"Candidate names : {candidate_names}")
            print(f"Canonical match : {canonical_matches}")
            print(f"Object keys      : {list(obj.keys())}")
            print("Object content preview:")
            pprint(obj, sort_dicts=False)

            # Limit printing to avoid too much output
            if matched_count >= 10:
                break

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()