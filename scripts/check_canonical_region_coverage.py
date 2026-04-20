# scripts/check_canonical_region_coverage.py

from pathlib import Path
import pandas as pd
import json
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"
REGION_CONFIG_PATH = PROJECT_ROOT / "configs" / "region_config_v1.json"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "canonical_region_coverage_v1.csv"


def load_scene_graph(scene_graph_path: Path):
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_raw_names(scene_graph: dict):
    names = []
    objects = scene_graph.get("objects", [])

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        for key in ["name", "label", "bbox_name", "object_name"]:
            if key in obj and isinstance(obj[key], str):
                cleaned = obj[key].strip().lower()
                if cleaned:
                    names.append(cleaned)

        for key in ["names", "labels"]:
            if key in obj and isinstance(obj[key], list):
                for x in obj[key]:
                    if isinstance(x, str):
                        cleaned = x.strip().lower()
                        if cleaned:
                            names.append(cleaned)

    return names


def main():
    print("=== CHECK CANONICAL REGION COVERAGE ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)

    with open(REGION_CONFIG_PATH, "r", encoding="utf-8") as f:
        region_config = json.load(f)

    canonical_regions = region_config["canonical_regions"]
    raw_to_canonical = {
        k.strip().lower(): v for k, v in region_config["raw_to_canonical"].items()
    }

    region_presence_counter = Counter()
    split_presence_counter = []

    for row in df.itertuples(index=False):
        scene_graph = load_scene_graph(Path(row.scene_graph_local_path))
        raw_names = extract_raw_names(scene_graph)

        canonical_found = set()
        for raw_name in raw_names:
            if raw_name in raw_to_canonical:
                canonical_found.add(raw_to_canonical[raw_name])

        for region in canonical_found:
            region_presence_counter[region] += 1

        split_presence_counter.append({
            "dicom_id": row.dicom_id,
            "official_split": row.official_split,
            **{region: int(region in canonical_found) for region in canonical_regions}
        })

    coverage_rows = []
    total_samples = len(df)

    for region in canonical_regions:
        count_present = region_presence_counter[region]
        coverage_rows.append({
            "canonical_region": region,
            "samples_with_region": count_present,
            "coverage_fraction": count_present / total_samples
        })

    coverage_df = pd.DataFrame(coverage_rows).sort_values(
        by="samples_with_region", ascending=False
    )
    coverage_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total working samples: {total_samples}")
    print("\nCanonical region coverage:")
    print(coverage_df)

    print(f"\nSaved coverage table to: {OUTPUT_CSV}")
    print("Canonical region coverage check completed successfully.")


if __name__ == "__main__":
    main()