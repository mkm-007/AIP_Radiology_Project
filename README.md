# Region-Guided Radiology Report Generation

Course project (CSC 8260 Advanced Image Processing, GSU): region-guided explainable chest X-ray report generation using anatomical boxes, retrieval, and generation baselines under leakage-safe grouped CV.

**Display title (locked):** Region-Guided Radiology Report Generation  
**Team:** Sai Kethan Bharadwaj Kanithi, Meghansh Siregey, Murali Krishna Maddineni

---

## Problem

Global-image report generators are hard to trust: clinicians cannot see *which anatomy* drove the text. We align **canonical thoracic regions** with report text and compare retrieval vs generation under grouped cross-validation.

## Build

Full course pipeline (needs MIMIC-CXR / Chest ImaGenome access — not redistributed here):

- Region crops from Chest ImaGenome boxes (heart, mediastinum, left/right lung)  
- ResNet18 region features → GRU decoder with region-id conditioning  
- Retrieval baseline + generation training/eval scripts under `scripts/`

**Offline demo** (this repo, no PHI/datasets):

```bash
python run_offline_demo.py
```

Synthesizes region feature vectors and a tiny retrieval/generation stub so the *story* is runnable without MIMIC.

## Proof (enrolled report metrics)

| Setting | Metric | Value |
|---------|--------|-------|
| Retrieval CV (best) | Jaccard | **0.1635 ± 0.0099** |
| Generation CV (best) | BLEU-1 | **0.3511 ± 0.0206** |
| Generation CV (best) | ROUGE-L | **0.4177 ± 0.0195** |
| Generation held-out test | BLEU-1 / BLEU-4 / ROUGE-L / METEOR | **0.306 / 0.127 / 0.381 / 0.290** |
| Retrieval held-out test | Jaccard | **0.114** |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python run_offline_demo.py
```

Full modeling (if you have licensed data locally): see `scripts/` and `README_project_scope.txt`.

## Honesty

Coursework research prototype. Not a clinical device. Metrics above are from the submitted AIP final report. Offline demo does **not** reproduce those numbers — it proves the region-guided loop structure only.
