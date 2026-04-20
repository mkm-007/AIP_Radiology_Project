# scripts/tune_generation_cv_v1.py

from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEN_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "generation_tokenized_dataset_v1.csv"
VOCAB_PATH       = PROJECT_ROOT / "data" / "processed" / "generation_vocab_v1.json"
FEATURES_PATH    = PROJECT_ROOT / "data" / "processed" / "region_features_resnet18_v1.npy"
FEATURE_META_PATH= PROJECT_ROOT / "data" / "processed" / "region_features_metadata_v1.csv"
CV_DATASET_PATH  = PROJECT_ROOT / "data" / "processed" / "modeling_dataset_with_cv_folds_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH   = OUTPUT_DIR / "generation_cv_fold_results_v1.csv"
SUMMARY_PATH        = OUTPUT_DIR / "generation_cv_summary_v1.csv"
BEST_CONFIG_PATH    = OUTPUT_DIR / "generation_cv_best_config_v1.csv"

DEVICE       = "cpu"
NUM_EPOCHS   = 6
BATCH_SIZE   = 32
NUM_WORKERS  = 0
MAX_DECODE   = 30
EMBED_DIM    = 128
REGION_EMBED = 32

CONFIGS = [
    {"lr": 1e-3, "hidden_dim": 256},
    {"lr": 5e-4, "hidden_dim": 256},
    {"lr": 1e-3, "hidden_dim": 128},
    {"lr": 5e-4, "hidden_dim": 128},
]

def parse_ids(s):
    return [int(x) for x in str(s).strip().split() if str(x).strip()]

def tokenize(text):
    return re.findall(r"[a-z0-9]+|[.,!?;:()-]", str(text).strip().lower())

def rouge_l_f1(ref, pred):
    r, p = tokenize(ref), tokenize(pred)
    m, n = len(r), len(p)
    if m == 0 or n == 0: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if r[i]==p[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    pr, rc = lcs/n, lcs/m
    return 2*pr*rc/(pr+rc) if pr+rc>0 else 0.0

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu1(ref, pred):
    r, p = tokenize(ref), tokenize(pred)
    if not p: return 0.0
    return float(sentence_bleu([r], p, weights=(1,0,0,0), smoothing_function=SmoothingFunction().method1))

class GenDS(Dataset):
    def __init__(self, df, features, region_to_id, pad_id):
        self.df, self.features, self.r2i, self.pad = df.reset_index(drop=True), features, region_to_id, pad_id
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = torch.tensor(self.features[int(row["feature_row_index"])], dtype=torch.float32)
        rid  = torch.tensor(self.r2i[row["canonical_region"]], dtype=torch.long)
        ids  = torch.tensor(parse_ids(row["target_token_ids"]), dtype=torch.long)
        return feat, rid, ids, row["region_text_target_normalized"]

def collate(batch, pad_id):
    feats = torch.stack([x[0] for x in batch])
    rids  = torch.stack([x[1] for x in batch])
    seqs  = [x[2] for x in batch]
    texts = [x[3] for x in batch]
    maxl  = max(len(s) for s in seqs)
    padded = torch.full((len(batch), maxl), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs): padded[i, :len(s)] = s
    return feats, rids, padded, texts

class GRUGen(nn.Module):
    def __init__(self, feat_dim, num_regions, vocab_size, hidden_dim, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.region_emb = nn.Embedding(num_regions, REGION_EMBED)
        self.token_emb  = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)
        self.init_mlp   = nn.Sequential(
            nn.Linear(feat_dim + REGION_EMBED, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.decoder     = nn.GRU(EMBED_DIM + REGION_EMBED, hidden_dim, batch_first=True)
        self.out         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, feats, rids, inp):
        rv  = self.region_emb(rids)
        h   = self.init_mlp(torch.cat([feats, rv], 1)).unsqueeze(0)
        te  = self.token_emb(inp)
        rr  = rv.unsqueeze(1).expand(-1, inp.size(1), -1)
        o,_ = self.decoder(torch.cat([te, rr], 2), h)
        return self.out(o)

    def greedy(self, feats, rids, bos, eos, maxl):
        rv  = self.region_emb(rids)
        h   = self.init_mlp(torch.cat([feats, rv], 1)).unsqueeze(0)
        cur = torch.full((feats.size(0),1), bos, dtype=torch.long)
        done= torch.zeros(feats.size(0), dtype=torch.bool)
        out = []
        for _ in range(maxl):
            te = self.token_emb(cur)
            rr = rv.unsqueeze(1)
            o, h = self.decoder(torch.cat([te, rr], 2), h)
            nx = torch.argmax(self.out(o[:,-1,:]), 1)
            out.append(nx); cur = nx.unsqueeze(1)
            done |= (nx == eos)
            if done.all(): break
        return torch.stack(out, 1) if out else torch.empty((feats.size(0),0), dtype=torch.long)

def decode(ids, i2t, eos, specials):
    toks = []
    for i in ids:
        i = int(i)
        if i == eos: break
        t = i2t.get(str(i), "<unk>")
        if t not in specials: toks.append(t)
    return " ".join(toks)

def run_fold(train_df, valid_df, features, vocab_obj, region_to_id, lr, hidden_dim):
    pad_id = vocab_obj["token_to_id"]["<pad>"]
    bos_id = vocab_obj["token_to_id"]["<bos>"]
    eos_id = vocab_obj["token_to_id"]["<eos>"]
    vocab_size = vocab_obj["vocab_size"]
    i2t = vocab_obj["id_to_token"]
    specials = {"<pad>","<bos>","<eos>","<unk>"}

    col_fn = lambda b: collate(b, pad_id)
    train_ld = DataLoader(GenDS(train_df, features, region_to_id, pad_id), BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=col_fn)
    valid_ld = DataLoader(GenDS(valid_df, features, region_to_id, pad_id), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=col_fn)

    model = GRUGen(features.shape[1], len(region_to_id), vocab_size, hidden_dim, pad_id)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_loss, best_bleu1, best_rouge = math.inf, 0.0, 0.0

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        for feats, rids, padded, _ in train_ld:
            logits = model(feats, rids, padded[:,:-1])
            loss   = crit(logits.reshape(-1, vocab_size), padded[:,1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        bleu1_scores, rouge_scores = [], []
        with torch.no_grad():
            for feats, rids, _, texts in valid_ld:
                gen = model.greedy(feats, rids, bos_id, eos_id, MAX_DECODE).numpy()
                for i, gt in enumerate(texts):
                    pred = decode(gen[i], i2t, eos_id, specials)
                    bleu1_scores.append(bleu1(gt, pred))
                    rouge_scores.append(rouge_l_f1(gt, pred))

        mb = float(np.mean(bleu1_scores))
        mr = float(np.mean(rouge_scores))
        if mb + mr > best_bleu1 + best_rouge:
            best_bleu1, best_rouge = mb, mr

    return best_bleu1, best_rouge

def main():
    print("=== TUNE GENERATION CV V1 ===")

    gen_df   = pd.read_csv(GEN_DATASET_PATH)
    feat_meta= pd.read_csv(FEATURE_META_PATH)
    features = np.load(FEATURES_PATH).astype(np.float32)
    cv_df    = pd.read_csv(CV_DATASET_PATH)

    with open(VOCAB_PATH) as f: vocab_obj = json.load(f)

    merge_cols = ["subject_id","study_id","dicom_id","official_split","canonical_region","crop_local_path"]
    merged = gen_df.merge(feat_meta[merge_cols+["feature_row_index"]], on=merge_cols, how="inner")
    merged = merged.merge(cv_df[merge_cols[:5]+["cv_fold"]], on=merge_cols[:5], how="left")

    train_rows = merged[merged["official_split"]=="train"].copy().reset_index(drop=True)
    region_to_id = {r:i for i,r in enumerate(sorted(train_rows["canonical_region"].unique()))}

    print(f"Train rows for CV: {len(train_rows)}")

    fold_results = []

    for cfg in CONFIGS:
        lr, hd = cfg["lr"], cfg["hidden_dim"]
        print(f"\nConfig: lr={lr}, hidden_dim={hd}")
        fold_bleus, fold_rouges = [], []

        for fold in sorted(train_rows["cv_fold"].dropna().unique()):
            fold = int(fold)
            tr = train_rows[train_rows["cv_fold"]!=fold].reset_index(drop=True)
            vl = train_rows[train_rows["cv_fold"]==fold].reset_index(drop=True)
            b1, rl = run_fold(tr, vl, features, vocab_obj, region_to_id, lr, hd)
            fold_bleus.append(b1); fold_rouges.append(rl)
            print(f"  Fold {fold}: bleu1={b1:.4f}, rouge_l={rl:.4f}")
            fold_results.append({"lr":lr,"hidden_dim":hd,"fold":fold,"bleu1":b1,"rouge_l":rl})

        print(f"  Mean bleu1={np.mean(fold_bleus):.4f}±{np.std(fold_bleus):.4f}  rouge_l={np.mean(fold_rouges):.4f}±{np.std(fold_rouges):.4f}")

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(FOLD_RESULTS_PATH, index=False)

    summary = (fold_df.groupby(["lr","hidden_dim"])
               .agg(bleu1_mean=("bleu1","mean"), bleu1_std=("bleu1","std"),
                    rouge_l_mean=("rouge_l","mean"), rouge_l_std=("rouge_l","std"))
               .reset_index()
               .sort_values("bleu1_mean", ascending=False))
    summary.to_csv(SUMMARY_PATH, index=False)

    best = summary.iloc[[0]].copy()
    best.to_csv(BEST_CONFIG_PATH, index=False)

    print("\n=== CV SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== BEST CONFIG ===")
    print(best.to_string(index=False))
    print(f"\nSaved: {FOLD_RESULTS_PATH}\n       {SUMMARY_PATH}\n       {BEST_CONFIG_PATH}")

if __name__ == "__main__":
    main()