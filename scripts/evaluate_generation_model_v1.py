# scripts/evaluate_generation_model_v1.py

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEN_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "generation_tokenized_dataset_v1.csv"
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "generation_vocab_v1.json"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
FEATURE_META_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "generation_model_v1_best.pt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV_PATH = OUTPUT_DIR / "generation_eval_predictions_v1.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "generation_eval_summary_v1.csv"
OVERALL_CSV_PATH = OUTPUT_DIR / "generation_eval_overall_v1.csv"

EMBED_DIM = 128
REGION_EMBED_DIM = 32
HIDDEN_DIM = 256
MAX_DECODE_LEN = 30
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cpu"


def parse_token_ids(s: str):
    return [int(x) for x in str(s).strip().split() if str(x).strip()]


def tokenize_text(text: str):
    text = str(text).strip().lower()
    return re.findall(r"[a-z0-9]+|[.,!?;:()-]", text)


def detokenize_token_list(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DummyWordNet:
    def synsets(self, word):
        return []


class GenerationEvalDataset(Dataset):
    def __init__(self, df, features, region_to_id):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.region_to_id = region_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = self.features[int(row["feature_row_index"])].astype(np.float32)
        region_id = self.region_to_id[row["canonical_region"]]

        return {
            "feature": torch.tensor(feat, dtype=torch.float32),
            "region_id": torch.tensor(region_id, dtype=torch.long),
            "target_text": row["region_text_target_normalized"],
            "canonical_region": row["canonical_region"],
            "dicom_id": row["dicom_id"],
            "official_split": row["official_split"],
        }


def collate_eval(batch):
    features = torch.stack([x["feature"] for x in batch], dim=0)
    region_ids = torch.stack([x["region_id"] for x in batch], dim=0)
    meta = {
        "target_text": [x["target_text"] for x in batch],
        "canonical_region": [x["canonical_region"] for x in batch],
        "dicom_id": [x["dicom_id"] for x in batch],
        "official_split": [x["official_split"] for x in batch],
    }
    return features, region_ids, meta


class RegionSentenceGenerator(nn.Module):
    def __init__(self, feature_dim, num_regions, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id

        self.region_emb = nn.Embedding(num_regions, REGION_EMBED_DIM)
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)

        self.init_mlp = nn.Sequential(
            nn.Linear(feature_dim + REGION_EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

        self.decoder = nn.GRU(
            input_size=EMBED_DIM + REGION_EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True
        )

        self.output_layer = nn.Linear(HIDDEN_DIM, vocab_size)

    def greedy_decode(self, features, region_ids, bos_id, eos_id, max_len):
        region_vec = self.region_emb(region_ids)
        hidden = self.init_mlp(torch.cat([features, region_vec], dim=1)).unsqueeze(0)

        B = features.size(0)
        current = torch.full((B, 1), bos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(B, dtype=torch.bool, device=features.device)

        generated = []

        for _ in range(max_len):
            tok_emb = self.token_emb(current)
            region_rep = region_vec.unsqueeze(1)
            decoder_in = torch.cat([tok_emb, region_rep], dim=2)

            out, hidden = self.decoder(decoder_in, hidden)
            logits = self.output_layer(out[:, -1, :])
            next_token = torch.argmax(logits, dim=1)

            generated.append(next_token)
            current = next_token.unsqueeze(1)

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=features.device)

        return torch.stack(generated, dim=1)


def decode_ids_to_text(id_seq, id_to_token, eos_id, special_token_set):
    toks = []
    for idx in id_seq:
        idx = int(idx)
        if idx == eos_id:
            break
        tok = id_to_token.get(str(idx), "<unk>")
        if tok in special_token_set:
            continue
        toks.append(tok)
    return detokenize_token_list(toks)


def exact_match(a: str, b: str) -> int:
    return int(str(a).strip().lower() == str(b).strip().lower())


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    m, n = len(ref_tokens), len(pred_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu_scores(reference: str, prediction: str):
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    if len(pred_tokens) == 0:
        return 0.0, 0.0

    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return float(bleu1), float(bleu4)


def meteor_safe(reference: str, prediction: str) -> float:
    ref_tokens = tokenize_text(reference)
    pred_tokens = tokenize_text(prediction)

    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    try:
        return float(meteor_score([ref_tokens], pred_tokens, wordnet=DummyWordNet()))
    except Exception:
        return 0.0


def main():
    print("=== EVALUATE GENERATION MODEL V1 ===")

    gen_df = pd.read_csv(GEN_DATASET_PATH)
    feat_meta = pd.read_csv(FEATURE_META_PATH)
    features = np.load(FEATURES_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    token_to_id = vocab_obj["token_to_id"]
    id_to_token = vocab_obj["id_to_token"]

    pad_id = token_to_id["<pad>"]
    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]
    vocab_size = int(vocab_obj["vocab_size"])

    region_to_id = checkpoint["region_to_id"]

    print(f"Generation dataset rows: {len(gen_df)}")
    print(f"Feature metadata rows  : {len(feat_meta)}")
    print(f"Feature shape          : {features.shape}")
    print(f"Vocab size             : {vocab_size}")

    merge_cols = [
        "subject_id", "study_id", "dicom_id",
        "official_split", "canonical_region", "crop_local_path"
    ]

    merged = gen_df.merge(
        feat_meta[merge_cols + ["feature_row_index"]],
        on=merge_cols,
        how="inner"
    )

    if len(merged) != len(gen_df):
        raise ValueError("Merged generation dataset row count does not match expected rows.")

    eval_df = merged[merged["official_split"].isin(["valid", "test"])].copy().reset_index(drop=True)
    print(f"Eval rows: {len(eval_df)}")

    eval_ds = GenerationEvalDataset(eval_df, features, region_to_id)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_eval
    )

    model = RegionSentenceGenerator(
        feature_dim=checkpoint["feature_dim"],
        num_regions=len(region_to_id),
        vocab_size=checkpoint["vocab_size"],
        pad_id=pad_id
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    special_token_set = {"<pad>", "<bos>", "<eos>", "<unk>"}
    prediction_rows = []

    with torch.no_grad():
        for features_batch, region_ids_batch, meta in eval_loader:
            features_batch = features_batch.to(DEVICE)
            region_ids_batch = region_ids_batch.to(DEVICE)

            generated = model.greedy_decode(
                features_batch,
                region_ids_batch,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=MAX_DECODE_LEN
            ).cpu().numpy()

            for i in range(len(meta["dicom_id"])):
                pred_text = decode_ids_to_text(
                    generated[i],
                    id_to_token=id_to_token,
                    eos_id=eos_id,
                    special_token_set=special_token_set
                )

                gt_text = str(meta["target_text"][i])

                bleu1, bleu4 = bleu_scores(gt_text, pred_text)
                rouge_l = rouge_l_f1(gt_text, pred_text)
                meteor = meteor_safe(gt_text, pred_text)

                prediction_rows.append({
                    "official_split": meta["official_split"][i],
                    "canonical_region": meta["canonical_region"][i],
                    "dicom_id": meta["dicom_id"][i],
                    "ground_truth_text": gt_text,
                    "generated_text": pred_text,
                    "exact_match": exact_match(gt_text, pred_text),
                    "bleu1": bleu1,
                    "bleu4": bleu4,
                    "rouge_l_f1": rouge_l,
                    "meteor": meteor,
                })

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

    summary_df = (
        pred_df.groupby(["official_split", "canonical_region"])
        .agg(
            row_count=("dicom_id", "count"),
            exact_match_rate=("exact_match", "mean"),
            mean_bleu1=("bleu1", "mean"),
            mean_bleu4=("bleu4", "mean"),
            mean_rouge_l_f1=("rouge_l_f1", "mean"),
            mean_meteor=("meteor", "mean"),
        )
        .reset_index()
        .sort_values(["official_split", "canonical_region"])
    )
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    overall_df = (
        pred_df.groupby(["official_split"])
        .agg(
            row_count=("dicom_id", "count"),
            exact_match_rate=("exact_match", "mean"),
            mean_bleu1=("bleu1", "mean"),
            mean_bleu4=("bleu4", "mean"),
            mean_rouge_l_f1=("rouge_l_f1", "mean"),
            mean_meteor=("meteor", "mean"),
        )
        .reset_index()
    )
    overall_df.to_csv(OVERALL_CSV_PATH, index=False)

    print("\nSummary by split and region:")
    print(summary_df)

    print("\nOverall summary:")
    print(overall_df)

    print("\nSample predictions:")
    print(pred_df[[
        "official_split", "canonical_region",
        "ground_truth_text", "generated_text",
        "bleu1", "bleu4", "rouge_l_f1", "meteor"
    ]].head(10))

    print("\n=== SAVED OUTPUTS ===")
    print(f"Predictions CSV : {PREDICTIONS_CSV_PATH}")
    print(f"Summary CSV     : {SUMMARY_CSV_PATH}")
    print(f"Overall CSV     : {OVERALL_CSV_PATH}")
    print("Generation evaluation completed successfully.")


if __name__ == "__main__":
    main()