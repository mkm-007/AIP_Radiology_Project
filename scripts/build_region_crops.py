# scripts/build_region_crops.py

from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGION_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "region_manifest_v1.csv"

CROP_ROOT = PROJECT_ROOT / "data" / "processed" / "region_crops_v1"
OUTPUT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "region_crop_manifest_v1.csv"
SUMMARY_CSV_PATH = PROJECT_ROOT / "results" / "tables" / "region_crop_summary_v1.csv"
PREVIEW_DIR = PROJECT_ROOT / "results" / "figures" / "region_crop_preview"

CROP_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

PREVIEW_PER_REGION = 2


def clip_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(1, min(int(x2), img_w))
    y2 = max(1, min(int(y2), img_h))

    # enforce valid crop box
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


def make_crop_filename(dicom_id, canonical_region):
    safe_region = canonical_region.replace("/", "_").replace(" ", "_")
    return f"{dicom_id}__{safe_region}.png"


def save_preview_image(full_img, crop_img, canonical_region, dicom_id, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(full_img, cmap="gray")
    axes[0].set_title("Full image")
    axes[0].axis("off")

    axes[1].imshow(crop_img, cmap="gray")
    axes[1].set_title(f"Crop: {canonical_region}")
    axes[1].axis("off")

    plt.suptitle(dicom_id, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== BUILD REGION CROPS ===")

    df = pd.read_csv(REGION_MANIFEST_PATH)
    print(f"Loaded region manifest rows: {len(df)}")

    output_rows = []
    preview_counter = {}

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        image_path = Path(row.image_local_path)
        canonical_region = row.canonical_region
        official_split = row.official_split
        dicom_id = row.dicom_id

        img = Image.open(image_path).convert("L")
        img_w, img_h = img.size

        x1, y1, x2, y2 = clip_bbox(
            row.original_x1,
            row.original_y1,
            row.original_x2,
            row.original_y2,
            img_w,
            img_h
        )

        crop = img.crop((x1, y1, x2, y2))

        crop_dir = CROP_ROOT / official_split / canonical_region
        crop_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = make_crop_filename(dicom_id, canonical_region)
        crop_path = crop_dir / crop_filename
        crop.save(crop_path)

        output_rows.append({
            **row._asdict(),
            "crop_local_path": str(crop_path.resolve()),
            "crop_width": crop.size[0],
            "crop_height": crop.size[1],
            "used_x1": x1,
            "used_y1": y1,
            "used_x2": x2,
            "used_y2": y2
        })

        # save a few preview figures only
        preview_counter.setdefault(canonical_region, 0)
        if preview_counter[canonical_region] < PREVIEW_PER_REGION:
            preview_path = PREVIEW_DIR / f"{canonical_region}_{preview_counter[canonical_region] + 1}.png"
            save_preview_image(
                full_img=img,
                crop_img=crop,
                canonical_region=canonical_region,
                dicom_id=dicom_id,
                save_path=preview_path
            )
            preview_counter[canonical_region] += 1

        if idx % 250 == 0 or idx == len(df):
            print(f"Processed {idx}/{len(df)} region rows")

    crop_df = pd.DataFrame(output_rows)
    crop_df.to_csv(OUTPUT_MANIFEST_PATH, index=False)

    summary_df = (
        crop_df.groupby(["official_split", "canonical_region"])
        .agg(
            row_count=("dicom_id", "count"),
            mean_crop_width=("crop_width", "mean"),
            mean_crop_height=("crop_height", "mean")
        )
        .reset_index()
        .sort_values(["official_split", "canonical_region"])
    )
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Crop manifest : {OUTPUT_MANIFEST_PATH}")
    print(f"Summary table : {SUMMARY_CSV_PATH}")
    print(f"Crop root     : {CROP_ROOT}")
    print(f"Preview dir   : {PREVIEW_DIR}")
    print("Region crop build completed successfully.")


if __name__ == "__main__":
    main()