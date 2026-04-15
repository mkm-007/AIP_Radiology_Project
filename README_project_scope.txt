Project Title:
Region-Guided Explainable Radiology Report Generation from Chest X-Rays

One-Week Build Scope:
- Use a reduced subset of the dataset
- Use existing anatomical region boxes from Chest ImaGenome
- Use chest X-ray images from MIMIC-CXR-JPG
- Use associated reports from MIMIC-CXR
- Build a region-guided baseline
- Extract region-level image features
- Generate region-level text targets
- Train and evaluate a manageable baseline
- Include cross-validation, hyperparameter tuning, mean ± std reporting, t-SNE, tables, and qualitative figures

Not in initial build:
- Full-scale reproduction of the target paper
- Detector training from scratch
- Prior-scan temporal modeling
- Large-scale end-to-end production system