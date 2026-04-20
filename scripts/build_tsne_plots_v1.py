# scripts/build_tsne_plots_v1.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "tsne"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TSNE_COORDS_CSV = TABLES_DIR / "tsne_coordinates_v1.csv"
TSNE_REGION_PNG = OUTPUT_DIR / "tsne_by_region_v1.png"
TSNE_SPLIT_PNG = OUTPUT_DIR / "tsne_by_split_v1.png"

PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def main():
    print("=== BUILD TSNE PLOTS V1 ===")

    features = np.load(FEATURES_PATH)
    meta = pd.read_csv(METADATA_PATH)

    print(f"Loaded feature array shape: {features.shape}")
    print(f"Loaded metadata rows      : {len(meta)}")

    if len(features) != len(meta):
        raise ValueError("Feature rows and metadata rows do not match.")

    # normalize + PCA first for faster/stabler t-SNE
    features = normalize_rows(features)

    pca_dim = min(PCA_DIM, features.shape[0], features.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    features_pca = pca.fit_transform(features)

    print(f"PCA output shape: {features_pca.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto"
    )

    coords = tsne.fit_transform(features_pca)

    tsne_df = meta.copy()
    tsne_df["tsne_x"] = coords[:, 0]
    tsne_df["tsne_y"] = coords[:, 1]
    tsne_df.to_csv(TSNE_COORDS_CSV, index=False)

    print(f"Saved coordinates CSV: {TSNE_COORDS_CSV}")

    # Plot 1: by canonical region
    plt.figure(figsize=(9, 7))
    for region in sorted(tsne_df["canonical_region"].unique()):
        sub = tsne_df[tsne_df["canonical_region"] == region]
        plt.scatter(sub["tsne_x"], sub["tsne_y"], s=18, alpha=0.75, label=region)

    plt.title("t-SNE of Region Features by Canonical Region")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TSNE_REGION_PNG, dpi=180)
    plt.close()

    # Plot 2: by official split
    plt.figure(figsize=(9, 7))
    for split in ["train", "valid", "test"]:
        sub = tsne_df[tsne_df["official_split"] == split]
        if len(sub) == 0:
            continue
        plt.scatter(sub["tsne_x"], sub["tsne_y"], s=18, alpha=0.75, label=split)

    plt.title("t-SNE of Region Features by Official Split")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TSNE_SPLIT_PNG, dpi=180)
    plt.close()

    print("\n=== SAVED OUTPUTS ===")
    print(f"t-SNE coordinates : {TSNE_COORDS_CSV}")
    print(f"Region plot       : {TSNE_REGION_PNG}")
    print(f"Split plot        : {TSNE_SPLIT_PNG}")
    print("t-SNE plot build completed successfully.")


if __name__ == "__main__":
    main()