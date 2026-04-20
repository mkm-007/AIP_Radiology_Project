# scripts/build_generation_vocab_and_tokens_v1.py

from pathlib import Path
import pandas as pd
import json
import re
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "generation_dataset_normalized_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZED_DATASET_PATH = OUTPUT_DIR / "generation_tokenized_dataset_v1.csv"
VOCAB_JSON_PATH = OUTPUT_DIR / "generation_vocab_v1.json"
SUMMARY_CSV_PATH = TABLES_DIR / "generation_tokenization_summary_v1.csv"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize_text(text: str):
    text = str(text).strip().lower()
    # keep simple punctuation tokens
    tokens = re.findall(r"[a-z0-9]+|[.,!?;:()-]", text)
    return tokens


def main():
    print("=== BUILD GENERATION VOCAB + TOKENS V1 ===")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded normalized generation rows: {len(df)}")

    if "region_text_target_normalized" not in df.columns:
        raise ValueError("Missing column: region_text_target_normalized")

    # Tokenize all rows
    df["target_tokens"] = df["region_text_target_normalized"].apply(tokenize_text)
    df["target_token_count"] = df["target_tokens"].apply(len)

    # -----------------------------
    # Build vocab from TRAIN only
    # -----------------------------
    train_df = df[df["official_split"] == "train"].copy()
    print(f"Training rows used for vocab: {len(train_df)}")

    counter = Counter()
    for toks in train_df["target_tokens"]:
        counter.update(toks)

    vocab_tokens = SPECIAL_TOKENS + sorted(counter.keys())
    token_to_id = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    print(f"Vocabulary size: {len(token_to_id)}")

    # -----------------------------
    # Encode all rows
    # -----------------------------
    encoded_sequences = []
    unk_counts = []

    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]
    unk_id = token_to_id["<unk>"]

    for toks in df["target_tokens"]:
        ids = [bos_id]
        unk_count = 0

        for tok in toks:
            if tok in token_to_id:
                ids.append(token_to_id[tok])
            else:
                ids.append(unk_id)
                unk_count += 1

        ids.append(eos_id)

        encoded_sequences.append(" ".join(map(str, ids)))
        unk_counts.append(unk_count)

    df["target_token_ids"] = encoded_sequences
    df["target_unk_count"] = unk_counts
    df["target_seq_len_with_specials"] = df["target_token_ids"].apply(lambda s: len(str(s).split()))

    # Save tokenized dataset
    df.to_csv(TOKENIZED_DATASET_PATH, index=False)

    # Save vocab JSON
    vocab_obj = {
        "special_tokens": SPECIAL_TOKENS,
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "vocab_size": len(token_to_id),
        "max_train_token_count": int(train_df["target_token_count"].max()),
        "mean_train_token_count": float(train_df["target_token_count"].mean())
    }

    with open(VOCAB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_obj, f, indent=2)

    # Summary table
    summary_rows = []
    for split_name in ["train", "valid", "test"]:
        sub = df[df["official_split"] == split_name].copy()
        if len(sub) == 0:
            continue

        summary_rows.append({
            "official_split": split_name,
            "row_count": len(sub),
            "mean_token_count": sub["target_token_count"].mean(),
            "max_token_count": sub["target_token_count"].max(),
            "rows_with_unk": int((sub["target_unk_count"] > 0).sum()),
            "mean_unk_count": sub["target_unk_count"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print("\nSummary by split:")
    print(summary_df)

    print("\nSample tokenized rows:")
    sample_df = df[[
        "canonical_region",
        "region_text_target_normalized",
        "target_token_ids"
    ]].head(5)

    for _, row in sample_df.iterrows():
        print(f"- [{row['canonical_region']}]")
        print(f"  text : {row['region_text_target_normalized']}")
        print(f"  ids  : {row['target_token_ids']}")

    print("\n=== SAVED OUTPUTS ===")
    print(f"Tokenized dataset : {TOKENIZED_DATASET_PATH}")
    print(f"Vocab JSON        : {VOCAB_JSON_PATH}")
    print(f"Summary CSV       : {SUMMARY_CSV_PATH}")
    print("Generation vocabulary + tokenization completed successfully.")


if __name__ == "__main__":
    main()