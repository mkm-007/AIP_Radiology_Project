from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
manifest = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "region_manifest_v1.csv")
out_dir = PROJECT_ROOT / "results" / "figures" / "bbox_examples"
out_dir.mkdir(parents=True, exist_ok=True)

COLORS = {"heart":"red","mediastinum":"blue","left_lung":"green","right_lung":"orange",
          "left_hemidiaphragm":"purple","right_hemidiaphragm":"brown"}

studies = manifest["dicom_id"].unique()
random.seed(42)
chosen = random.sample(list(studies), min(4, len(studies)))

for dicom_id in chosen:
    sub = manifest[manifest["dicom_id"]==dicom_id]
    img_path = Path(sub.iloc[0]["image_local_path"])
    if not img_path.exists(): continue
    img = Image.open(img_path).convert("L")
    w, h = img.size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")

    for _, row in sub.iterrows():
        region = row["canonical_region"]
        x1 = float(row["original_x1"]); y1 = float(row["original_y1"])
        x2 = float(row["original_x2"]); y2 = float(row["original_y2"])
        color = COLORS.get(region, "yellow")
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1-5,0), region, color=color, fontsize=8, fontweight="bold")

    ax.axis("off")
    ax.set_title(f"Anatomical Regions — {dicom_id[:20]}", fontsize=10)
    plt.tight_layout()
    save_path = out_dir / f"bbox_{dicom_id[:20]}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

print("Done.")