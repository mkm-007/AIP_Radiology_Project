# scripts/sanity_check_dataset.py

from pathlib import Path
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "working_manifest_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "sanity_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5
RANDOM_SEED = 42


def read_report_text(report_path: Path) -> str:
    with open(report_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_scene_graph(scene_graph_path: Path) -> dict:
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_region_info(scene_graph: dict):
    objects = scene_graph.get("objects", [])
    return objects, len(objects)


def make_preview_figure(image_path: Path, report_text: str, num_regions: int, save_path: Path):
    img = Image.open(image_path).convert("L")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    wrapped_report = "\n".join(textwrap.wrap(report_text[:400], width=80))
    title_text = f"Preview\nRegions in scene graph: {num_regions}\n\nReport excerpt:\n{wrapped_report}"
    ax.set_title(title_text, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== DATASET SANITY CHECK ===")

    df = pd.read_csv(WORKING_MANIFEST_PATH)
    print(f"Loaded working manifest rows: {len(df)}")

    sample_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=RANDOM_SEED).copy()

    summary_rows = []

    for idx, row in enumerate(sample_df.itertuples(index=False), start=1):
        image_path = Path(row.image_local_path)
        report_path = Path(row.report_local_path)
        scene_graph_path = Path(row.scene_graph_local_path)

        print("\n" + "=" * 80)
        print(f"SAMPLE {idx}")
        print("=" * 80)
        print(f"subject_id     : {row.subject_id}")
        print(f"study_id       : {row.study_id}")
        print(f"dicom_id       : {row.dicom_id}")
        print(f"official_split : {row.official_split}")
        print(f"view_position  : {row.ViewPosition}")
        print(f"image_path     : {image_path}")
        print(f"report_path    : {report_path}")
        print(f"scene_graph    : {scene_graph_path}")

        # Load files
        img = Image.open(image_path)
        report_text = read_report_text(report_path)
        scene_graph = read_scene_graph(scene_graph_path)
        objects, num_regions = extract_region_info(scene_graph)

        print(f"image_size     : {img.size}")
        print(f"report_chars   : {len(report_text)}")
        print(f"scene_objects  : {num_regions}")
        print(f"scene_keys     : {list(scene_graph.keys())}")

        preview_path = OUTPUT_DIR / f"sample_{idx}_{row.dicom_id}.png"
        make_preview_figure(
            image_path=image_path,
            report_text=report_text,
            num_regions=num_regions,
            save_path=preview_path
        )

        summary_rows.append({
            "sample_index": idx,
            "subject_id": row.subject_id,
            "study_id": row.study_id,
            "dicom_id": row.dicom_id,
            "official_split": row.official_split,
            "view_position": row.ViewPosition,
            "image_width": img.size[0],
            "image_height": img.size[1],
            "report_chars": len(report_text),
            "scene_object_count": num_regions,
            "preview_path": str(preview_path.resolve())
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_DIR / "sanity_check_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETED")
    print("=" * 80)
    print(f"Saved preview images to: {OUTPUT_DIR}")
    print(f"Saved summary CSV to   : {summary_csv_path}")


if __name__ == "__main__":
    main()