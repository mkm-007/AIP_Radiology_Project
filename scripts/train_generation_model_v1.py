# scripts/train_generation_model_v1.py

from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEN_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "generation_tokenized_dataset_v1.csv"
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "generation_vocab_v1.json"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
FEATURE_META_PATH = PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "generation_model_v1_best.pt"
HISTORY_CSV_PATH = TABLES_DIR / "generation_train_history_v1.csv"
VALID_SAMPLES_CSV_PATH = TABLES_DIR / "generation_valid_samples_v1.csv"
SUMMARY_CSV_PATH = TABLES_DIR / "generation_train_summary_v1.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 12
LEARNING_RATE = 1e-3
EMBED_DIM = 128
REGION_EMBED_DIM = 32
HIDDEN_DIM = 256
MAX_DECODE_LEN = 30
PATIENCE = 3
NUM_WORKERS = 0  # safest on Windows
DEVICE = "cpu"


def parse_token_ids(s: str):
    return [int(x) for x in str(s).strip().split() if str(x).strip()]


def detokenize_token_list(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\\1", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class GenerationDataset(Dataset):
    def __init__(self, df, features, region_to_id):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.region_to_id = region_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_row_index = int(row["feature_row_index"])
        feat = self.features[feature_row_index].astype(np.float32)

        region_id = self.region_to_id[row["canonical_region"]]
        seq_ids = parse_token_ids(row["target_token_ids"])

        return {
            "feature": torch.tensor(feat, dtype=torch.float32),
            "region_id": torch.tensor(region_id, dtype=torch.long),
            "seq_ids": torch.tensor(seq_ids, dtype=torch.long),
            "target_text": row["region_text_target_normalized"],
            "canonical_region": row["canonical_region"],
            "dicom_id": row["dicom_id"],
            "official_split": row["official_split"],
        }


def collate_batch(batch, pad_id):
    features = torch.stack([x["feature"] for x in batch], dim=0)
    region_ids = torch.stack([x["region_id"] for x in batch], dim=0)

    seqs = [x["seq_ids"] for x in batch]
    max_len = max(len(s) for s in seqs)

    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s

    meta = {
        "target_text": [x["target_text"] for x in batch],
        "canonical_region": [x["canonical_region"] for x in batch],
        "dicom_id": [x["dicom_id"] for x in batch],
        "official_split": [x["official_split"] for x in batch],
    }

    return features, region_ids, padded, meta


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

    def forward(self, features, region_ids, input_tokens):
        """
        features: [B, 512]
        region_ids: [B]
        input_tokens: [B, T]
        """
        region_vec = self.region_emb(region_ids)  # [B, R]
        init_hidden = self.init_mlp(torch.cat([features, region_vec], dim=1)).unsqueeze(0)  # [1,B,H]

        tok_emb = self.token_emb(input_tokens)  # [B,T,E]
        region_rep = region_vec.unsqueeze(1).expand(-1, input_tokens.size(1), -1)  # [B,T,R]
        decoder_in = torch.cat([tok_emb, region_rep], dim=2)  # [B,T,E+R]

        out, _ = self.decoder(decoder_in, init_hidden)  # [B,T,H]
        logits = self.output_layer(out)  # [B,T,V]
        return logits

    def greedy_decode(self, features, region_ids, bos_id, eos_id, max_len):
        region_vec = self.region_emb(region_ids)
        hidden = self.init_mlp(torch.cat([features, region_vec], dim=1)).unsqueeze(0)

        B = features.size(0)
        current = torch.full((B, 1), bos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(B, dtype=torch.bool, device=features.device)

        generated = []

        for _ in range(max_len):
            tok_emb = self.token_emb(current)  # [B,1,E]
            region_rep = region_vec.unsqueeze(1)  # [B,1,R]
            decoder_in = torch.cat([tok_emb, region_rep], dim=2)

            out, hidden = self.decoder(decoder_in, hidden)
            logits = self.output_layer(out[:, -1, :])  # [B,V]
            next_token = torch.argmax(logits, dim=1)  # [B]

            generated.append(next_token)
            current = next_token.unsqueeze(1)

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=features.device)

        return torch.stack(generated, dim=1)  # [B,T]


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


def run_epoch(model, loader, optimizer, criterion, pad_id, bos_id, eos_id, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    all_pred_rows = []

    with torch.set_grad_enabled(train_mode):
        for features, region_ids, padded, meta in loader:
            features = features.to(DEVICE)
            region_ids = region_ids.to(DEVICE)
            padded = padded.to(DEVICE)

            input_tokens = padded[:, :-1]
            target_tokens = padded[:, 1:]

            logits = model(features, region_ids, input_tokens)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_mask = (target_tokens != pad_id)
            num_valid = int(valid_mask.sum().item())
            total_loss += float(loss.item()) * max(num_valid, 1)
            total_tokens += max(num_valid, 1)

            if not train_mode:
                generated = model.greedy_decode(
                    features, region_ids,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=MAX_DECODE_LEN
                ).cpu().numpy()

                for i in range(len(meta["dicom_id"])):
                    all_pred_rows.append({
                        "official_split": meta["official_split"][i],
                        "canonical_region": meta["canonical_region"][i],
                        "dicom_id": meta["dicom_id"][i],
                        "ground_truth_text": meta["target_text"][i],
                        "generated_token_ids": " ".join(map(str, generated[i].tolist()))
                    })

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, all_pred_rows


def main():
    print("=== TRAIN GENERATION MODEL V1 ===")

    gen_df = pd.read_csv(GEN_DATASET_PATH)
    feat_meta = pd.read_csv(FEATURE_META_PATH)
    features = np.load(FEATURES_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)

    token_to_id = vocab_obj["token_to_id"]
    id_to_token = vocab_obj["id_to_token"]

    pad_id = token_to_id["<pad>"]
    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]
    vocab_size = int(vocab_obj["vocab_size"])

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

    train_df = merged[merged["official_split"] == "train"].copy().reset_index(drop=True)
    valid_df = merged[merged["official_split"] == "valid"].copy().reset_index(drop=True)

    print(f"Train rows: {len(train_df)}")
    print(f"Valid rows: {len(valid_df)}")

    region_names = sorted(train_df["canonical_region"].unique())
    region_to_id = {r: i for i, r in enumerate(region_names)}

    train_ds = GenerationDataset(train_df, features, region_to_id)
    valid_ds = GenerationDataset(valid_df, features, region_to_id)

    collate_fn = lambda batch: collate_batch(batch, pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = RegionSentenceGenerator(
        feature_dim=features.shape[1],
        num_regions=len(region_to_id),
        vocab_size=vocab_size,
        pad_id=pad_id
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    history_rows = []
    best_valid_loss = math.inf
    best_epoch = -1
    patience_counter = 0
    best_valid_predictions = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, _ = run_epoch(
            model, train_loader, optimizer, criterion,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
            train_mode=True
        )

        valid_loss, valid_pred_rows = run_epoch(
            model, valid_loader, optimizer, criterion,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
            train_mode=False
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        })

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            patience_counter = 0
            best_valid_predictions = valid_pred_rows

            torch.save({
                "model_state_dict": model.state_dict(),
                "region_to_id": region_to_id,
                "vocab_size": vocab_size,
                "feature_dim": features.shape[1],
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
            }, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)

    # Decode best valid predictions
    special_token_set = {"<pad>", "<bos>", "<eos>", "<unk>"}
    decoded_rows = []

    if best_valid_predictions is not None:
        for row in best_valid_predictions:
            pred_ids = parse_token_ids(row["generated_token_ids"])
            pred_text = decode_ids_to_text(
                pred_ids, id_to_token=id_to_token,
                eos_id=eos_id,
                special_token_set=special_token_set
            )

            decoded_rows.append({
                "official_split": row["official_split"],
                "canonical_region": row["canonical_region"],
                "dicom_id": row["dicom_id"],
                "ground_truth_text": row["ground_truth_text"],
                "generated_text": pred_text
            })

    valid_pred_df = pd.DataFrame(decoded_rows)
    valid_pred_df.to_csv(VALID_SAMPLES_CSV_PATH, index=False)

    summary_df = pd.DataFrame([{
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "vocab_size": vocab_size,
        "num_regions": len(region_to_id),
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "embed_dim": EMBED_DIM,
        "region_embed_dim": REGION_EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
    }])
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print("\n=== SAVED OUTPUTS ===")
    print(f"Best model checkpoint : {BEST_MODEL_PATH}")
    print(f"Training history CSV  : {HISTORY_CSV_PATH}")
    print(f"Valid samples CSV     : {VALID_SAMPLES_CSV_PATH}")
    print(f"Summary CSV           : {SUMMARY_CSV_PATH}")
    print(f"Best epoch            : {best_epoch}")
    print(f"Best valid loss       : {best_valid_loss:.4f}")
    print("Generation model training completed successfully.")


if __name__ == "__main__":
    main()