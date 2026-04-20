# scripts/extract_region_features_v1.py

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_NPY = OUTPUT_DIR / "region_features_resnet18_v1.npy"
METADATA_CSV = OUTPUT_DIR / "region_features_metadata_v1.csv"
SUMMARY_CSV = TABLES_DIR / "region_feature_extraction_summary_v1.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0  # safest on Windows
IMAGE_SIZE = 224


class RegionCropDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["crop_local_path"])

        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, idx


def build_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Remove final classification layer, keep pooled feature vector (512-dim)
    model.fc = torch.nn.Identity()
    model.eval()

    return model, weights


def main():
    print("=== EXTRACT REGION FEATURES V1 ===")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded modeling dataset rows: {len(df)}")

    model, weights = build_model()
    preprocess = weights.transforms()

    dataset = RegionCropDataset(df, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_features = np.zeros((len(df), 512), dtype=np.float32)

    with torch.no_grad():
        for batch_idx, (images, row_indices) in enumerate(loader, start=1):
            feats = model(images)   # shape: [B, 512]
            feats = feats.cpu().numpy()

            for i, row_idx in enumerate(row_indices.tolist()):
                all_features[row_idx] = feats[i]

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"Processed batch {batch_idx}/{len(loader)}")

    np.save(FEATURES_NPY, all_features)

    metadata_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "official_split",
        "canonical_region",
        "crop_local_path",
        "region_text_target",
        "target_char_count"
    ]
    metadata_df = df[metadata_cols].copy()
    metadata_df["feature_row_index"] = np.arange(len(metadata_df))
    metadata_df.to_csv(METADATA_CSV, index=False)

    summary_df = pd.DataFrame([{
        "num_rows": len(df),
        "feature_dim": all_features.shape[1],
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "feature_file": str(FEATURES_NPY),
        "metadata_file": str(METADATA_CSV)
    }])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Feature array : {FEATURES_NPY}")
    print(f"Metadata CSV  : {METADATA_CSV}")
    print(f"Summary CSV   : {SUMMARY_CSV}")
    print(f"Feature shape : {all_features.shape}")
    print("Region feature extraction completed successfully.")


if __name__ == "__main__":
    main()